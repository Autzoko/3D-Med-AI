"""
SegMamba + Mask2Former 完整训练脚本 - 修复版

主要修复：
1. ✅ Deep Supervision权重修复（使用衰减权重，避免累加过高）
2. ✅ 添加真实Dice Score计算和监控
3. ✅ 改进的训练输出（显示Loss和Score）
4. ✅ 基于Dice Score保存最佳模型
5. ✅ 更合理的学习率和优化器设置
6. ✅ Hungarian Matching保留（完整Mask2Former）

参考：
- Mask2Former官方实现
- SegMamba训练代码
- 用户反馈的问题修复

作者：基于用户代码修复
日期：2025-12-05
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

# 导入模型和数据
from model.model import segmamba_mask2former_tiny, segmamba_mask2former_small, segmamba_mask2former_base
from data.dataloader import NPYSegmentationDataset, get_default_transforms


# ============================================================================
# Hungarian Matcher (保持不变，已验证正确)
# ============================================================================

class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for bipartite matching
    参考：Mask2Former官方 modeling/matcher.py
    """
    
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
        """
        Args:
            outputs: dict with 'pred_logits' (B, Q, C) and 'pred_masks' (B, Q, H, W)
            targets: dict with 'labels' List[(N,)] and 'masks' List[(N, H, W)]
        
        Returns:
            List of (src_idx, tgt_idx) tuples
        """
        bs = outputs["pred_logits"].shape[0]
        
        # Flatten
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # (B*Q, C)
        out_mask = outputs["pred_masks"].flatten(0, 1)  # (B*Q, H, W)
        
        # Concatenate targets
        tgt_ids = torch.cat([v for v in targets["labels"]])
        tgt_mask = torch.cat([v for v in targets["masks"]])
        
        # 1. Classification cost
        cost_class = -out_prob[:, tgt_ids]
        
        # 2. Mask cost (point sampling)
        out_mask_flat = out_mask.flatten(1)  # (B*Q, H*W)
        tgt_mask_flat = tgt_mask.flatten(1).float()  # (N, H*W)
        
        num_points = min(self.num_points, out_mask_flat.shape[1])
        point_idx = torch.randperm(out_mask_flat.shape[1], device=out_mask.device)[:num_points]
        
        out_mask_sampled = out_mask_flat[:, point_idx].sigmoid()  # (B*Q, P)
        tgt_mask_sampled = tgt_mask_flat[:, point_idx]  # (N, P)
        
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
# Set Criterion (修复版 - 关键改进)
# ============================================================================

class SetCriterion(nn.Module):
    """
    Set Criterion for Mask2Former - 修复版
    
    关键修复：
    1. Deep Supervision使用衰减权重
    2. 添加真实Dice Score计算
    """
    
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float = 0.1,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        
        # Background weight
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with keys
                - pred_logits: (B, Q, C)
                - pred_masks: (B, Q, H, W)
                - aux_outputs: List of dicts (auxiliary predictions)
            targets: dict with keys
                - labels: List of (N,) tensors
                - masks: List of (N, H, W) tensors
        """
        # Matching
        indices = self.matcher(outputs, targets)
        
        # Compute losses for final layer
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_masks(outputs, targets, indices, self.count_masks(targets)))
        
        # Deep Supervision
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_indices = self.matcher(aux_outputs, targets)
                l_dict = self.loss_labels(aux_outputs, targets, aux_indices)
                l_dict.update(self.loss_masks(aux_outputs, targets, aux_indices, self.count_masks(targets)))
                l_dict = {f'{k}_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        
        return losses
    
    def count_masks(self, targets):
        """Count total number of masks"""
        return sum([len(t) for t in targets["labels"]])
    
    def loss_labels(self, outputs, targets, indices):
        """Classification loss"""
        pred_logits = outputs["pred_logits"]  # (B, Q, C)
        
        # Prepare target classes
        target_classes = torch.full(
            pred_logits.shape[:2],
            self.num_classes,
            dtype=torch.long,
            device=pred_logits.device
        )
        
        # Fill in matched classes
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[batch_idx, src_idx] = targets["labels"][batch_idx][tgt_idx]
        
        # Cross entropy loss
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),
            target_classes,
            self.empty_weight,
            reduction='mean'
        )
        
        return {"loss_ce": loss_ce}
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Mask BCE + Dice loss"""
        pred_masks = outputs["pred_masks"]  # (B, Q, H, W)
        
        # Collect matched predictions and targets
        src_masks = []
        tgt_masks = []
        
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            src_masks.append(pred_masks[batch_idx, src_idx])
            tgt_masks.append(targets["masks"][batch_idx][tgt_idx])
        
        if len(src_masks) == 0:
            # No masks
            return {
                "loss_mask": pred_masks.sum() * 0.0,
                "loss_dice": pred_masks.sum() * 0.0,
            }
        
        src_masks = torch.cat(src_masks)  # (N, H, W)
        tgt_masks = torch.cat(tgt_masks).float()  # (N, H, W)
        
        # Point sampling
        point_coords = self.get_point_coords_with_randomness(
            src_masks.unsqueeze(1),  # (N, 1, H, W)
            self.calculate_uncertainty,
            self.num_points,
            self.oversample_ratio,
            self.importance_sample_ratio,
        )
        
        # Sample points
        point_logits = self.point_sample(
            src_masks.unsqueeze(1),
            point_coords,
            align_corners=False
        ).squeeze(1)  # (N, P)
        
        point_labels = self.point_sample(
            tgt_masks.unsqueeze(1),
            point_coords,
            align_corners=False
        ).squeeze(1)  # (N, P)
        
        # BCE loss
        loss_mask = F.binary_cross_entropy_with_logits(
            point_logits,
            point_labels,
            reduction="mean"
        )
        
        # Dice loss
        loss_dice = self.dice_loss(
            point_logits.sigmoid(),
            point_labels,
            num_masks
        )
        
        losses = {
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
        }
        return losses
    
    def dice_loss(self, inputs, targets, num_masks):
        """Dice loss"""
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        
        return loss.sum() / num_masks
    
    def calculate_uncertainty(self, logits):
        """Uncertainty estimation"""
        uncertainty = -(logits * logits.sigmoid()).sum(1)
        return uncertainty
    
    def get_point_coords_with_randomness(
        self,
        coarse_logits,
        uncertainty_func,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
    ):
        """Point sampling with importance sampling"""
        num_boxes = coarse_logits.shape[0]
        num_sampled = int(num_points * oversample_ratio)
        
        # Sample uniformly
        point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
        point_logits = self.point_sample(coarse_logits, point_coords, align_corners=False)
        
        # Calculate uncertainty
        point_uncertainties = uncertainty_func(point_logits)
        
        # 修复：处理维度
        if point_uncertainties.dim() == 3:
            point_uncertainties = point_uncertainties.squeeze(1)
        
        # Importance sampling
        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points
        
        idx = torch.topk(point_uncertainties, k=num_uncertain_points, dim=1)[1]
        shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
        idx += shift[:, None]
        
        point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
        
        if num_random_points > 0:
            random_point_coords = torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device)
            point_coords = torch.cat([point_coords, random_point_coords], dim=1)
        
        return point_coords
    
    def point_sample(self, input, point_coords, **kwargs):
        """Point sampling"""
        add_dim = False
        if point_coords.dim() == 3:
            add_dim = True
            point_coords = point_coords.unsqueeze(2)
        
        # Convert coordinates
        point_coords = 2.0 * point_coords - 1.0
        output = F.grid_sample(input, point_coords, **kwargs)
        
        if add_dim:
            output = output.squeeze(3)
        
        return output


# ============================================================================
# Helper Functions
# ============================================================================

def prepare_targets(batch: Dict) -> Dict:
    """
    Convert semantic masks to instance format
    """
    masks = batch['mask']  # (B, H, W)
    B, H, W = masks.shape
    device = masks.device
    
    labels_list = []
    masks_list = []
    
    for i in range(B):
        mask = masks[i]
        unique_classes = torch.unique(mask)
        unique_classes = unique_classes[unique_classes != 0]  # Remove background
        
        if len(unique_classes) == 0:
            # No foreground, create dummy target
            labels_list.append(torch.tensor([0], dtype=torch.long, device=device))
            masks_list.append(torch.zeros((1, H, W), dtype=torch.float32, device=device))
        else:
            instance_masks = []
            instance_labels = []
            for cls in unique_classes:
                binary_mask = (mask == cls).float()
                instance_masks.append(binary_mask)
                instance_labels.append(cls.item() - 1)  # 0-indexed for model
            
            labels_list.append(torch.tensor(instance_labels, dtype=torch.long, device=device))
            masks_list.append(torch.stack(instance_masks))
    
    return {
        "labels": labels_list,
        "masks": masks_list,
    }


@torch.no_grad()
def calculate_dice_score(pred_masks, pred_logits, gt_masks, threshold=0.5):
    """
    计算真实的Dice Score（评估指标）
    
    Args:
        pred_masks: (B, Q, H, W)
        pred_logits: (B, Q, C)
        gt_masks: (B, H, W)
    
    Returns:
        dice_score: float [0, 1]
    """
    B, Q, H, W = pred_masks.shape
    
    # 选择confidence最高的query（非背景类）
    # pred_logits: (B, Q, C), C = num_classes + 1
    # 取前num_classes的最大confidence
    confidences = pred_logits.softmax(dim=-1)[..., :-1].max(dim=-1)[0]  # (B, Q)
    best_queries = confidences.argmax(dim=1)  # (B,)
    
    # 收集最佳预测
    pred_masks_binary = []
    for i in range(B):
        best_mask = pred_masks[i, best_queries[i]]  # (H, W)
        pred_masks_binary.append((best_mask.sigmoid() > threshold).float())
    
    pred_masks_binary = torch.stack(pred_masks_binary)  # (B, H, W)
    gt_masks_binary = (gt_masks > 0).float()  # (B, H, W)
    
    # 计算Dice
    intersection = (pred_masks_binary * gt_masks_binary).sum()
    union = pred_masks_binary.sum() + gt_masks_binary.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection + 1) / (union + 1)
    
    return dice.item()



# ============================================================================
# Training and Validation Functions
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    criterion: SetCriterion,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scheduler=None,
) -> Dict[str, float]:
    """训练一个epoch"""
    
    model.train()
    criterion.train()
    
    total_loss = 0
    loss_dict_accumulated = {}
    total_dice = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Prepare targets
        targets = prepare_targets({'mask': masks})
        
        # Forward
        outputs = model(images)
        
        # Compute loss
        loss_dict = criterion(outputs, targets)
        
        # Weighted sum
        losses = sum(
            loss_dict[k] * criterion.weight_dict[k]
            for k in loss_dict.keys()
            if k in criterion.weight_dict
        )
        
        # Backward
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # 计算Dice Score
        with torch.no_grad():
            dice = calculate_dice_score(
                outputs['pred_masks'],
                outputs['pred_logits'],
                masks
            )
            total_dice += dice
        
        # Accumulate loss
        total_loss += losses.item()
        for k, v in loss_dict.items():
            if k not in loss_dict_accumulated:
                loss_dict_accumulated[k] = 0
            loss_dict_accumulated[k] += v.item()
        
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'dice': f'{dice:.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}',
            'avg_dice': f'{total_dice / num_batches:.4f}'
        })
    
    # Calculate averages
    avg_losses = {k: v / num_batches for k, v in loss_dict_accumulated.items()}
    avg_losses['total_loss'] = total_loss / num_batches
    avg_losses['dice_score'] = total_dice / num_batches  # 新增！
    
    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: SetCriterion,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """验证"""
    
    model.eval()
    criterion.eval()
    
    total_loss = 0
    loss_dict_accumulated = {}
    total_dice = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Validating')
    
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        targets = prepare_targets({'mask': masks})
        
        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        
        losses = sum(
            loss_dict[k] * criterion.weight_dict[k]
            for k in loss_dict.keys()
            if k in criterion.weight_dict
        )
        
        # 计算Dice Score
        dice = calculate_dice_score(
            outputs['pred_masks'],
            outputs['pred_logits'],
            masks
        )
        
        total_loss += losses.item()
        total_dice += dice
        
        for k, v in loss_dict.items():
            if k not in loss_dict_accumulated:
                loss_dict_accumulated[k] = 0
            loss_dict_accumulated[k] += v.item()
        
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'dice': f'{dice:.4f}'
        })
    
    avg_losses = {k: v / num_batches for k, v in loss_dict_accumulated.items()}
    avg_losses['total_loss'] = total_loss / num_batches
    avg_losses['dice_score'] = total_dice / num_batches  # 新增！
    
    return avg_losses


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SegMamba + Mask2Former Training - FIXED VERSION')
    
    # Data
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=1)
    
    # Model
    parser.add_argument('--model_size', type=str, default='small', choices=['tiny', 'small', 'base'])
    parser.add_argument('--num_queries', type=int, default=20)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)  # 降低学习率
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Matcher
    parser.add_argument('--cost_class', type=float, default=2.0)
    parser.add_argument('--cost_mask', type=float, default=5.0)
    parser.add_argument('--cost_dice', type=float, default=5.0)
    
    # Loss weights
    parser.add_argument('--class_weight', type=float, default=2.0)
    parser.add_argument('--mask_weight', type=float, default=5.0)
    parser.add_argument('--dice_weight', type=float, default=5.0)
    parser.add_argument('--eos_coef', type=float, default=0.1)
    parser.add_argument('--aux_weight', type=float, default=0.4)  # 新增：aux层权重衰减
    
    # Point sampling
    parser.add_argument('--num_points', type=int, default=12544)
    parser.add_argument('--oversample_ratio', type=float, default=3.0)
    parser.add_argument('--importance_sample_ratio', type=float, default=0.75)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--save_every', type=int, default=10)
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("="*80)
    print("SegMamba + Mask2Former Training - FIXED VERSION")
    print("="*80)
    print(f"Device: {device}")
    
    # =========================================================================
    # [1] DataLoaders
    # =========================================================================
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
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    # =========================================================================
    # [2] Model
    # =========================================================================
    print(f"\n[2] Creating model...")
    
    if args.model_size == 'tiny':
        model = segmamba_mask2former_tiny(
            num_classes=args.num_classes,
            num_queries=args.num_queries
        )
    elif args.model_size == 'small':
        model = segmamba_mask2former_small(
            num_classes=args.num_classes,
            num_queries=args.num_queries
        )
    else:
        model = segmamba_mask2former_base(
            num_classes=args.num_classes,
            num_queries=args.num_queries
        )
    
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {args.model_size}")
    print(f"  Parameters: {num_params / 1e6:.2f}M")
    
    # =========================================================================
    # [3] Criterion - 关键修复：使用衰减权重
    # =========================================================================
    print(f"\n[3] Creating criterion...")
    
    matcher = HungarianMatcher(
        cost_class=args.cost_class,
        cost_mask=args.cost_mask,
        cost_dice=args.cost_dice,
        num_points=args.num_points,
    )
    
    # 主层权重
    weight_dict = {
        "loss_ce": args.class_weight,
        "loss_mask": args.mask_weight,
        "loss_dice": args.dice_weight,
    }
    
    # 辅助层权重 - 使用衰减！
    num_decoder_layers = model.get_num_layers()
    print(f"  Decoder layers: {num_decoder_layers}")
    print(f"  Main layer weights: CE={args.class_weight}, Mask={args.mask_weight}, Dice={args.dice_weight}")
    print(f"  Aux layer weight decay: {args.aux_weight}")
    
    for i in range(num_decoder_layers - 1):
        weight_dict.update({
            f"loss_ce_{i}": args.class_weight * args.aux_weight,
            f"loss_mask_{i}": args.mask_weight * args.aux_weight,
            f"loss_dice_{i}": args.dice_weight * args.aux_weight,
        })
    
    # 计算预期总权重
    main_weight = args.class_weight + args.mask_weight + args.dice_weight
    aux_total_weight = (num_decoder_layers - 1) * main_weight * args.aux_weight
    total_weight = main_weight + aux_total_weight
    print(f"  Expected total loss weight: {total_weight:.2f}")
    print(f"    Main layer: {main_weight:.2f}")
    print(f"    Aux layers: {aux_total_weight:.2f}")
    
    criterion = SetCriterion(
        num_classes=args.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        num_points=args.num_points,
        oversample_ratio=args.oversample_ratio,
        importance_sample_ratio=args.importance_sample_ratio,
    )
    criterion = criterion.to(device)
    
    # =========================================================================
    # [4] Optimizer - 使用更保守的学习率
    # =========================================================================
    print(f"\n[4] Creating optimizer...")
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 使用OneCycleLR with warmup
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    print(f"  Optimizer: AdamW")
    print(f"  LR: {args.lr}")
    print(f"  Scheduler: OneCycleLR with 10% warmup")
    
    # =========================================================================
    # [5] Training Loop
    # =========================================================================
    print(f"\n[5] Starting training...")
    print("="*80)
    
    best_dice = 0.0
    train_losses_history = []
    val_losses_history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-"*80)
        
        # Train
        train_losses = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, scheduler
        )
        
        # Validate
        val_losses = validate(model, criterion, val_loader, device)
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        train_losses_history.append(train_losses)
        val_losses_history.append(val_losses)
        
        # Print results - 改进的输出格式
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_losses['total_loss']:.4f}")
        print(f"    - CE: {train_losses.get('loss_ce', 0):.4f}")
        print(f"    - Mask: {train_losses.get('loss_mask', 0):.4f}")
        print(f"    - Dice Loss: {train_losses.get('loss_dice', 0):.4f}")
        print(f"  Train Dice Score: {train_losses['dice_score']:.4f}")  # 新增！
        print(f"  Val Loss: {val_losses['total_loss']:.4f}")
        print(f"  Val Dice Score: {val_losses['dice_score']:.4f}")  # 新增！
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model based on Dice Score
        if val_losses['dice_score'] > best_dice:
            best_dice = val_losses['dice_score']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice_score': best_dice,
                'val_loss': val_losses['total_loss'],
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  ✓ New best model! Dice Score: {best_dice:.4f}")
        
        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice_score': val_losses['dice_score'],
                'val_loss': val_losses['total_loss'],
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))
            print(f"  ✓ Saved checkpoint: checkpoint_epoch_{epoch}.pth")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'dice_score': val_losses['dice_score'],
        'val_loss': val_losses['total_loss'],
    }, os.path.join(args.output_dir, 'final_model.pth'))
    
    # Save training history
    np.save(os.path.join(args.output_dir, 'train_losses.npy'), train_losses_history)
    np.save(os.path.join(args.output_dir, 'val_losses.npy'), val_losses_history)
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"  Best Dice Score: {best_dice:.4f}")
    print(f"  Output dir: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()