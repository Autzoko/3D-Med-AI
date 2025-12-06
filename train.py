"""
SegMamba + Mask2Former 完整训练脚本 - 索引修复版

只修复索引越界问题，保持原始所有结构
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment

from model.model import segmamba_mask2former_tiny, segmamba_mask2former_small, segmamba_mask2former_base
from data.dataloader import NPYSegmentationDataset, get_default_transforms


# ============================================================================
# Hungarian Matcher (只修复索引问题)
# ============================================================================

class HungarianMatcher(nn.Module):
    """Hungarian Matcher - 修复索引越界"""
    
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        num_points: int = 12544,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        bs = outputs["pred_logits"].shape[0]
        
        # Flatten
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # (B*Q, C+1)
        out_mask = outputs["pred_masks"].flatten(0, 1)  # (B*Q, H, W)
        
        # Concatenate targets
        tgt_ids = torch.cat([v for v in targets["labels"]])
        tgt_mask = torch.cat([v for v in targets["masks"]])
        
        # 🔧 修复1: 确保tgt_ids类型正确且在有效范围内
        num_classes = out_prob.shape[-1] - 1  # C (不包含background)
        tgt_ids = tgt_ids.long()  # 确保是long类型
        tgt_ids = torch.clamp(tgt_ids, 0, num_classes)  # 限制范围 [0, num_classes]
        
        # 1. Classification cost - 🔧 使用gather避免索引错误
        # 原来: cost_class = -out_prob[:, tgt_ids]  这会出错！
        # 正确做法: 使用gather
        tgt_ids_expanded = tgt_ids.unsqueeze(0).expand(out_prob.shape[0], -1)  # (B*Q, N)
        cost_class = -torch.gather(out_prob, 1, tgt_ids_expanded)  # (B*Q, N)
        
        # 2. Mask cost (point sampling)
        out_mask_flat = out_mask.flatten(1)  # (B*Q, H*W)
        tgt_mask_flat = tgt_mask.flatten(1).float()  # (N, H*W)
        
        num_points = min(self.num_points, out_mask_flat.shape[1])
        point_idx = torch.randperm(out_mask_flat.shape[1], device=out_mask.device)[:num_points]
        
        out_mask_sampled = out_mask_flat[:, point_idx].sigmoid()  # (B*Q, P)
        tgt_mask_sampled = tgt_mask_flat[:, point_idx]  # (N, P)
        
        # 🔧 修复2: 安全的BCE计算
        with torch.no_grad():
            cost_mask = F.binary_cross_entropy_with_logits(
                out_mask_flat[:, point_idx].unsqueeze(1).expand(-1, tgt_mask_sampled.shape[0], -1),
                tgt_mask_sampled.unsqueeze(0).expand(out_mask_sampled.shape[0], -1, -1),
                reduction='none'
            ).mean(-1)
        
        # 3. Dice cost
        numerator = 2 * (out_mask_sampled.unsqueeze(1) * tgt_mask_sampled.unsqueeze(0)).sum(-1)
        denominator = out_mask_sampled.unsqueeze(1).sum(-1) + tgt_mask_sampled.unsqueeze(0).sum(-1)
        cost_dice = 1 - (numerator + 1) / (denominator + 1)
        
        # Final cost
        C = self.cost_class * cost_class + self.cost_mask * cost_mask + self.cost_dice * cost_dice
        C = C.view(bs, -1, tgt_mask.shape[0]).cpu()
        
        # Hungarian matching
        sizes = [len(v) for v in targets["labels"]]
        indices = []
        offset = 0
        for i, size in enumerate(sizes):
            if size == 0:
                indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                continue
            c = C[i, :, offset:offset+size]
            src_idx, tgt_idx = linear_sum_assignment(c)
            indices.append((torch.as_tensor(src_idx, dtype=torch.long), torch.as_tensor(tgt_idx, dtype=torch.long)))
            offset += size
        
        return indices


# ============================================================================
# Set Criterion (保持原样)
# ============================================================================

class SetCriterion(nn.Module):
    """Set criterion with deep supervision"""
    
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef=0.1,
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(self, outputs, targets, indices):
        pred_logits = outputs["pred_logits"]
        
        target_classes = torch.full(
            pred_logits.shape[:2],
            self.num_classes,
            dtype=torch.long,
            device=pred_logits.device
        )
        
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                # 🔧 确保标签在有效范围内
                tgt_labels = targets["labels"][batch_idx][tgt_idx]
                tgt_labels = torch.clamp(tgt_labels.long(), 0, self.num_classes)
                target_classes[batch_idx, src_idx] = tgt_labels
        
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),
            target_classes,
            self.empty_weight,
            reduction='mean'
        )
        
        return {"loss_ce": loss_ce}
    
    def loss_masks(self, outputs, targets, indices):
        pred_masks = outputs["pred_masks"]
        
        src_masks = []
        target_masks = []
        
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                src_masks.append(pred_masks[batch_idx, src_idx])
                target_masks.append(targets["masks"][batch_idx][tgt_idx])
        
        if len(src_masks) == 0:
            return {
                "loss_mask": pred_masks.sum() * 0.0,
                "loss_dice": pred_masks.sum() * 0.0
            }
        
        src_masks = torch.cat(src_masks, dim=0)
        target_masks = torch.cat(target_masks, dim=0).float()
        
        # Point sampling
        num_total_points = int(self.num_points * self.oversample_ratio)
        point_coords = torch.rand(1, num_total_points, 2, device=src_masks.device)
        
        src_masks_points = self._sample_points(src_masks.unsqueeze(1), point_coords)
        target_masks_points = self._sample_points(target_masks.unsqueeze(1), point_coords)
        
        # Importance sampling
        with torch.no_grad():
            point_uncertainties = -(src_masks_points.abs() - 0.5).abs()
            num_uncertain_points = int(self.importance_sample_ratio * self.num_points)
            num_random_points = self.num_points - num_uncertain_points
            
            idx = torch.topk(point_uncertainties.squeeze(1), k=num_uncertain_points, dim=1)[1]
            shift = num_uncertain_points * torch.arange(src_masks_points.shape[0], dtype=torch.long, device=src_masks.device)
            idx += shift.unsqueeze(1)
            
            src_masks_points = src_masks_points.squeeze(1).flatten(0, 1)[idx].view(-1, num_uncertain_points)
            target_masks_points = target_masks_points.squeeze(1).flatten(0, 1)[idx].view(-1, num_uncertain_points)
            
            point_coords_rand = torch.rand(src_masks.shape[0], num_random_points, 2, device=src_masks.device)
            src_masks_points_rand = self._sample_points(src_masks.unsqueeze(1), point_coords_rand).squeeze(1)
            target_masks_points_rand = self._sample_points(target_masks.unsqueeze(1), point_coords_rand).squeeze(1)
            
            src_masks_points = torch.cat([src_masks_points, src_masks_points_rand], dim=1)
            target_masks_points = torch.cat([target_masks_points, target_masks_points_rand], dim=1)
        
        # Losses
        loss_mask = F.binary_cross_entropy_with_logits(src_masks_points, target_masks_points, reduction='mean')
        
        src_masks_sigmoid = src_masks_points.sigmoid()
        numerator = 2 * (src_masks_sigmoid * target_masks_points).sum(dim=1)
        denominator = src_masks_sigmoid.sum(dim=1) + target_masks_points.sum(dim=1) + 1e-4
        loss_dice = (1 - numerator / denominator).mean()
        
        return {
            "loss_mask": loss_mask,
            "loss_dice": loss_dice
        }
    
    def _sample_points(self, input, point_coords):
        input = input.float()
        point_coords = point_coords.float()
        
        add_dim = False
        if point_coords.dim() == 3:
            add_dim = True
            point_coords = point_coords.unsqueeze(2)
        
        output = F.grid_sample(input, 2.0 * point_coords - 1.0, align_corners=False)
        
        if add_dim:
            output = output.squeeze(3)
        
        return output
    
    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_masks(outputs, targets, indices))
        
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices_aux = self.matcher(aux_outputs, targets)
                losses_aux = {}
                losses_aux.update(self.loss_labels(aux_outputs, targets, indices_aux))
                losses_aux.update(self.loss_masks(aux_outputs, targets, indices_aux))
                losses.update({f"{k}_{i}": v for k, v in losses_aux.items()})
        
        return losses


# ============================================================================
# Training/Validation (保持原样)
# ============================================================================

def calculate_dice_score(pred_masks, gt_masks):
    pred_masks_binary = []
    for pred_mask in pred_masks:
        pred_binary = (pred_mask.sigmoid() > 0.5).float()
        pred_masks_binary.append(pred_binary)
    
    pred_masks_binary = torch.stack(pred_masks_binary)
    gt_masks_binary = (gt_masks > 0).float()
    
    if pred_masks_binary.shape[-2:] != gt_masks_binary.shape[-2:]:
        pred_masks_binary = F.interpolate(
            pred_masks_binary.unsqueeze(1),
            size=gt_masks_binary.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        pred_masks_binary = (pred_masks_binary > 0.5).float()
    
    intersection = (pred_masks_binary * gt_masks_binary).sum()
    union = pred_masks_binary.sum() + gt_masks_binary.sum()
    
    dice = (2.0 * intersection) / (union + 1e-8)
    
    return dice.item()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    
    loss_dict_accumulated = {}
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        labels = batch.get('label', None)
        
        if labels is None:
            labels = [torch.ones(1, dtype=torch.long, device=device) for _ in range(len(images))]
        else:
            labels = [l.to(device) for l in labels]
        
        targets = {
            "masks": [m.squeeze(0) for m in masks],
            "labels": labels
        }
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        losses = criterion(outputs, targets)
        
        weighted_losses = {}
        for k, v in losses.items():
            if k in criterion.weight_dict:
                weighted_losses[k] = v * criterion.weight_dict[k]
        
        loss = sum(weighted_losses.values())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        
        dice = calculate_dice_score(outputs["pred_masks"], masks)
        
        for k, v in losses.items():
            if k not in loss_dict_accumulated:
                loss_dict_accumulated[k] = 0.0
            loss_dict_accumulated[k] += v.item()
        
        total_loss += loss.item()
        total_dice += dice
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}'
        })
    
    avg_losses = {k: v / num_batches for k, v in loss_dict_accumulated.items()}
    avg_losses['total_loss'] = total_loss / num_batches
    avg_losses['dice_score'] = total_dice / num_batches
    
    return avg_losses


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    
    loss_dict_accumulated = {}
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc='Validation'):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        labels = batch.get('label', None)
        
        if labels is None:
            labels = [torch.ones(1, dtype=torch.long, device=device) for _ in range(len(images))]
        else:
            labels = [l.to(device) for l in labels]
        
        targets = {
            "masks": [m.squeeze(0) for m in masks],
            "labels": labels
        }
        
        outputs = model(images)
        
        losses = criterion(outputs, targets)
        
        weighted_losses = {}
        for k, v in losses.items():
            if k in criterion.weight_dict:
                weighted_losses[k] = v * criterion.weight_dict[k]
        
        loss = sum(weighted_losses.values())
        
        dice = calculate_dice_score(outputs["pred_masks"], masks)
        
        for k, v in losses.items():
            if k not in loss_dict_accumulated:
                loss_dict_accumulated[k] = 0.0
            loss_dict_accumulated[k] += v.item()
        
        total_loss += loss.item()
        total_dice += dice
        num_batches += 1
    
    avg_losses = {k: v / num_batches for k, v in loss_dict_accumulated.items()}
    avg_losses['total_loss'] = total_loss / num_batches
    avg_losses['dice_score'] = total_dice / num_batches
    
    return avg_losses


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SegMamba + Mask2Former Training - Fixed')
    
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=1)
    
    parser.add_argument('--model_size', type=str, default='small', choices=['tiny', 'small', 'base'])
    parser.add_argument('--num_queries', type=int, default=20)
    
    # 新增参数
    parser.add_argument('--drop_path_rate', type=float, default=0.0)
    parser.add_argument('--use_skip_connections', action='store_true')
    
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--cost_class', type=float, default=2.0)
    parser.add_argument('--cost_mask', type=float, default=5.0)
    parser.add_argument('--cost_dice', type=float, default=5.0)
    
    parser.add_argument('--class_weight', type=float, default=2.0)
    parser.add_argument('--mask_weight', type=float, default=5.0)
    parser.add_argument('--dice_weight', type=float, default=5.0)
    parser.add_argument('--eos_coef', type=float, default=0.1)
    parser.add_argument('--aux_weight', type=float, default=0.4)
    
    parser.add_argument('--num_points', type=int, default=12544)
    parser.add_argument('--oversample_ratio', type=float, default=3.0)
    parser.add_argument('--importance_sample_ratio', type=float, default=0.75)
    
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--save_every', type=int, default=10)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("="*80)
    print("SegMamba + Mask2Former Training")
    print("="*80)
    print(f"Device: {device}")
    
    # DataLoaders
    print("\n[1] Creating DataLoaders...")
    
    train_dataset = NPYSegmentationDataset(
        root_dir=args.data_root,
        split='train',
        image_size=args.image_size,
        transforms=get_default_transforms(args.image_size, 'train'),
        num_classes=args.num_classes,
        return_instance=False,
    )
    
    val_dataset = NPYSegmentationDataset(
        root_dir=args.data_root,
        split='val',
        image_size=args.image_size,
        transforms=get_default_transforms(args.image_size, 'val'),
        num_classes=args.num_classes,
        return_instance=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda x: {
            'image': torch.stack([item['image'] for item in x]),
            'mask': torch.stack([item['mask'] for item in x]),
            'label': [item.get('label', torch.tensor([1], dtype=torch.long)) for item in x]
        }
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda x: {
            'image': torch.stack([item['image'] for item in x]),
            'mask': torch.stack([item['mask'] for item in x]),
            'label': [item.get('label', torch.tensor([1], dtype=torch.long)) for item in x]
        }
    )
    
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    print("\n[2] Creating model...")
    
    if args.model_size == 'tiny':
        model = segmamba_mask2former_tiny(
            num_classes=args.num_classes,
            num_queries=args.num_queries,
            drop_path_rate=args.drop_path_rate,
            use_skip_connections=args.use_skip_connections
        )
    elif args.model_size == 'small':
        model = segmamba_mask2former_small(
            num_classes=args.num_classes,
            num_queries=args.num_queries,
            drop_path_rate=args.drop_path_rate,
            use_skip_connections=args.use_skip_connections
        )
    else:
        model = segmamba_mask2former_base(
            num_classes=args.num_classes,
            num_queries=args.num_queries,
            drop_path_rate=args.drop_path_rate,
            use_skip_connections=args.use_skip_connections
        )
    
    model = model.to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Criterion
    print("\n[3] Creating criterion...")
    
    matcher = HungarianMatcher(
        cost_class=args.cost_class,
        cost_mask=args.cost_mask,
        cost_dice=args.cost_dice,
        num_points=args.num_points,
    )
    
    weight_dict = {
        "loss_ce": args.class_weight,
        "loss_mask": args.mask_weight,
        "loss_dice": args.dice_weight,
    }
    
    num_layers = model.get_num_layers()
    for i in range(num_layers - 1):
        decay = args.aux_weight ** (num_layers - i - 1)
        weight_dict.update({
            f"loss_ce_{i}": args.class_weight * decay,
            f"loss_mask_{i}": args.mask_weight * decay,
            f"loss_dice_{i}": args.dice_weight * decay,
        })
    
    criterion = SetCriterion(
        num_classes=args.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        num_points=args.num_points,
        oversample_ratio=args.oversample_ratio,
        importance_sample_ratio=args.importance_sample_ratio,
    )
    
    # Optimizer
    print("\n[4] Creating optimizer...")
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=0.1)
    
    # Training
    print("\n[5] Training...")
    print("="*80)
    
    best_dice = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_losses = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_losses = validate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train: Loss {train_losses['total_loss']:.4f}, Dice {train_losses['dice_score']:.4f}")
        print(f"  Val:   Loss {val_losses['total_loss']:.4f}, Dice {val_losses['dice_score']:.4f}")
        
        if val_losses['dice_score'] > best_dice:
            best_dice = val_losses['dice_score']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_dice': best_dice,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  ✓ Best: {best_dice:.4f}")
        
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        scheduler.step()
    
    print(f"\n{'='*80}")
    print(f"Training completed! Best Dice: {best_dice:.4f}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()