"""
SegMamba + Mask2Former 训练脚本
完全符合Mask2Former论文实现

关键点:
1. Hungarian Matching - 二分图匹配
2. Deep Supervision - 所有decoder层都有loss
3. 三种loss: Classification + Mask BCE + Dice
4. Point-based sampling - 省内存

参考:
- Mask2Former论文: https://arxiv.org/abs/2112.01527
- 官方实现: https://github.com/facebookresearch/Mask2Former
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple
import json
from pathlib import Path

# 导入我们的模块
from data.dataloader import NPYSegmentationDataset, get_default_transforms
from model import segmamba_mask2former_small, segmamba_mask2former_tiny


# ============================================================================
# Hungarian Matcher - 完全按照Mask2Former实现
# ============================================================================

class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher用于计算预测和GT之间的最优匹配
    
    这是Mask2Former的核心组件，基于论文Section 3.1实现
    """
    
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        num_points: int = 12544,  # 论文中用于计算mask loss的采样点数
    ):
        """
        Args:
            cost_class: 分类cost的权重
            cost_mask: mask BCE cost的权重
            cost_dice: dice cost的权重
            num_points: 采样点数（用于计算mask cost）
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points
    
    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,  # (B, Q, num_classes+1)
        pred_masks: torch.Tensor,   # (B, Q, H, W)
        gt_labels: List[torch.Tensor],  # List of (N_i,)
        gt_masks: List[torch.Tensor],   # List of (N_i, H, W)
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        计算匹配
        
        Returns:
            List of (src_idx, tgt_idx) tuples
            src_idx: 预测的索引
            tgt_idx: GT的索引
        """
        batch_size, num_queries = pred_logits.shape[:2]
        
        # 将logits转为概率
        pred_probs = pred_logits.softmax(-1)  # (B, Q, num_classes+1)
        
        indices = []
        
        for b in range(batch_size):
            # 当前batch的预测和GT
            pred_prob = pred_probs[b]  # (Q, num_classes+1)
            pred_mask = pred_masks[b]  # (Q, H, W)
            gt_label = gt_labels[b]    # (N,)
            gt_mask = gt_masks[b]      # (N, H, W)
            
            num_gt = len(gt_label)
            
            if num_gt == 0:
                # 没有GT，不需要匹配
                indices.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long)
                ))
                continue
            
            # === 1. Classification Cost ===
            # 取出每个GT类别对应的预测概率
            cost_class = -pred_prob[:, gt_label]  # (Q, N)
            
            # === 2. Mask Cost ===
            # Point-based sampling（论文Section 3.3.2）
            # 随机采样点来计算mask cost，节省内存
            pred_mask_flat = pred_mask.flatten(1)  # (Q, H*W)
            gt_mask_flat = gt_mask.flatten(1).float()  # (N, H*W)
            
            # 采样点（如果点数少于num_points，使用所有点）
            num_points = min(self.num_points, pred_mask_flat.shape[1])
            point_idx = torch.randperm(pred_mask_flat.shape[1])[:num_points]
            
            pred_mask_sampled = pred_mask_flat[:, point_idx]  # (Q, num_points)
            gt_mask_sampled = gt_mask_flat[:, point_idx]      # (N, num_points)
            
            # Sigmoid后计算BCE cost
            pred_mask_sampled = pred_mask_sampled.sigmoid()
            
            # BCE cost: (Q, N)
            cost_mask = F.binary_cross_entropy_with_logits(
                pred_mask_sampled.unsqueeze(1).expand(-1, num_gt, -1),  # (Q, N, num_points)
                gt_mask_sampled.unsqueeze(0).expand(num_queries, -1, -1),
                reduction='none'
            ).mean(-1)
            
            # === 3. Dice Cost ===
            # Dice cost: (Q, N)
            numerator = 2 * (pred_mask_sampled.unsqueeze(1) * gt_mask_sampled.unsqueeze(0)).sum(-1)
            denominator = pred_mask_sampled.unsqueeze(1).sum(-1) + gt_mask_sampled.unsqueeze(0).sum(-1)
            cost_dice = 1 - (numerator + 1) / (denominator + 1)
            
            # === 总Cost ===
            cost = (
                self.cost_class * cost_class +
                self.cost_mask * cost_mask +
                self.cost_dice * cost_dice
            )  # (Q, N)
            
            # Hungarian算法求最优匹配
            cost = cost.cpu()
            src_idx, tgt_idx = linear_sum_assignment(cost)
            
            indices.append((
                torch.as_tensor(src_idx, dtype=torch.long),
                torch.as_tensor(tgt_idx, dtype=torch.long)
            ))
        
        return indices


# ============================================================================
# Mask2Former Loss - 完全按照论文实现
# ============================================================================

class Mask2FormerLoss(nn.Module):
    """
    Mask2Former完整Loss函数
    
    包含三个部分（论文Section 3.3）:
    1. Classification Loss
    2. Mask BCE Loss
    3. Dice Loss
    
    使用Deep Supervision训练所有decoder层
    """
    
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float] = None,
        eos_coef: float = 0.1,  # 背景类（no-object）的权重
        num_points: int = 12544,
    ):
        """
        Args:
            num_classes: 类别数（不含背景）
            matcher: Hungarian matcher
            weight_dict: loss权重字典
            eos_coef: 背景类权重
            num_points: 采样点数
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.num_points = num_points
        
        # 默认权重（来自Mask2Former论文）
        if weight_dict is None:
            self.weight_dict = {
                'loss_ce': 2.0,      # Classification
                'loss_mask': 5.0,    # Mask BCE
                'loss_dice': 5.0,    # Dice
            }
        else:
            self.weight_dict = weight_dict
        
        # 背景类权重调整
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef  # 最后一个是背景类
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(
        self,
        pred_logits: torch.Tensor,
        gt_labels: List[torch.Tensor],
        indices: List[Tuple],
    ) -> torch.Tensor:
        """
        Classification Loss
        """
        batch_size, num_queries = pred_logits.shape[:2]
        device = pred_logits.device
        
        # 创建目标标签（所有query默认为背景类）
        target_classes = torch.full(
            (batch_size, num_queries),
            self.num_classes,  # 背景类是最后一个
            dtype=torch.long,
            device=device
        )
        
        # 根据匹配结果填入真实标签
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[b, src_idx] = gt_labels[b][tgt_idx]
        
        # Cross Entropy Loss（使用背景类权重）
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),  # (B, num_classes+1, Q)
            target_classes,
            weight=self.empty_weight
        )
        
        return loss_ce
    
    def loss_masks(
        self,
        pred_masks: torch.Tensor,
        gt_masks: List[torch.Tensor],
        indices: List[Tuple],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mask BCE Loss + Dice Loss
        使用point-based sampling（论文Section 3.3.2）
        """
        batch_size = pred_masks.shape[0]
        device = pred_masks.device
        
        # 收集匹配的预测和GT
        src_masks = []
        target_masks = []
        
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                src_masks.append(pred_masks[b, src_idx])      # (N_matched, H, W)
                target_masks.append(gt_masks[b][tgt_idx])     # (N_matched, H, W)
        
        if len(src_masks) == 0:
            # 没有匹配的，返回0 loss
            return (
                pred_masks.sum() * 0.0,
                pred_masks.sum() * 0.0
            )
        
        # 拼接所有batch
        src_masks = torch.cat(src_masks, dim=0)      # (N_total, H, W)
        target_masks = torch.cat(target_masks, dim=0).float()
        
        # Flatten
        src_masks = src_masks.flatten(1)       # (N_total, H*W)
        target_masks = target_masks.flatten(1)
        
        # Point-based sampling
        num_points = min(self.num_points, src_masks.shape[1])
        point_idx = torch.randperm(src_masks.shape[1], device=device)[:num_points]
        
        src_masks = src_masks[:, point_idx]      # (N_total, num_points)
        target_masks = target_masks[:, point_idx]
        
        # === BCE Loss ===
        loss_mask = F.binary_cross_entropy_with_logits(
            src_masks,
            target_masks,
            reduction='mean'
        )
        
        # === Dice Loss ===
        src_masks = src_masks.sigmoid()
        numerator = 2 * (src_masks * target_masks).sum(1)
        denominator = src_masks.sum(1) + target_masks.sum(1)
        loss_dice = 1 - (numerator + 1) / (denominator + 1)
        loss_dice = loss_dice.mean()
        
        return loss_mask, loss_dice
    
    def forward(
        self,
        outputs: Dict,
        gt_labels: List[torch.Tensor],
        gt_masks: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        计算总loss
        
        Args:
            outputs: 模型输出，包含pred_logits, pred_masks, aux_outputs
            gt_labels: List of (N_i,) 类别标签
            gt_masks: List of (N_i, H, W) masks
        """
        # 最后一层的输出
        pred_logits = outputs['pred_logits']  # (B, Q, num_classes+1)
        pred_masks = outputs['pred_masks']    # (B, Q, H, W)
        
        # Hungarian Matching
        indices = self.matcher(pred_logits, pred_masks, gt_labels, gt_masks)
        
        # === 计算最后一层的loss ===
        loss_ce = self.loss_labels(pred_logits, gt_labels, indices)
        loss_mask, loss_dice = self.loss_masks(pred_masks, gt_masks, indices)
        
        losses = {
            'loss_ce': loss_ce * self.weight_dict['loss_ce'],
            'loss_mask': loss_mask * self.weight_dict['loss_mask'],
            'loss_dice': loss_dice * self.weight_dict['loss_dice'],
        }
        
        # === Deep Supervision: 所有中间层也计算loss ===
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_pred_logits = aux_outputs['pred_logits']
                aux_pred_masks = aux_outputs['pred_masks']
                
                # 重新匹配（每一层独立匹配）
                aux_indices = self.matcher(aux_pred_logits, aux_pred_masks, gt_labels, gt_masks)
                
                # 计算loss
                aux_loss_ce = self.loss_labels(aux_pred_logits, gt_labels, aux_indices)
                aux_loss_mask, aux_loss_dice = self.loss_masks(aux_pred_masks, gt_masks, aux_indices)
                
                losses[f'loss_ce_{i}'] = aux_loss_ce * self.weight_dict['loss_ce']
                losses[f'loss_mask_{i}'] = aux_loss_mask * self.weight_dict['loss_mask']
                losses[f'loss_dice_{i}'] = aux_loss_dice * self.weight_dict['loss_dice']
        
        return losses


# ============================================================================
# 准备实例格式的GT数据
# ============================================================================

def prepare_instance_targets(batch: Dict) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    将semantic mask转换为instance格式
    
    Args:
        batch: DataLoader返回的batch
        
    Returns:
        gt_labels: List of (N_i,) 类别标签
        gt_masks: List of (N_i, H, W) bool masks
    """
    masks = batch['mask']  # (B, H, W)
    batch_size = masks.shape[0]
    
    gt_labels = []
    gt_masks = []
    
    for b in range(batch_size):
        mask = masks[b]  # (H, W)
        
        # 提取实例
        unique_classes = torch.unique(mask)
        unique_classes = unique_classes[unique_classes > 0]  # 排除背景
        
        if len(unique_classes) == 0:
            # 没有前景，创建一个dummy instance
            gt_labels.append(torch.tensor([0], dtype=torch.long, device=mask.device))
            gt_masks.append(torch.zeros((1, *mask.shape), dtype=torch.bool, device=mask.device))
        else:
            instance_masks = []
            instance_labels = []
            
            for class_id in unique_classes:
                class_mask = (mask == class_id)
                instance_masks.append(class_mask)
                instance_labels.append(class_id - 1)  # 0-indexed（假设GT是1-indexed）
            
            gt_masks.append(torch.stack(instance_masks, dim=0))  # (N, H, W)
            gt_labels.append(torch.tensor(instance_labels, dtype=torch.long, device=mask.device))
    
    return gt_labels, gt_masks


# ============================================================================
# 训练函数
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    criterion: Mask2FormerLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """训练一个epoch"""
    
    model.train()
    criterion.train()
    
    total_loss = 0
    loss_dict_accumulated = {}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # 数据移到device
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # 准备GT（转为instance格式）
        gt_labels, gt_masks = prepare_instance_targets({'mask': masks})
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        
        # 计算loss
        losses = criterion(outputs, gt_labels, gt_masks)
        
        # 总loss
        loss = sum(losses.values())
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（可选，防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        
        optimizer.step()
        
        # 累积loss
        total_loss += loss.item()
        for k, v in losses.items():
            if k not in loss_dict_accumulated:
                loss_dict_accumulated[k] = 0
            loss_dict_accumulated[k] += v.item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
    
    # 计算平均loss
    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in loss_dict_accumulated.items()}
    avg_losses['total_loss'] = total_loss / num_batches
    
    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: Mask2FormerLoss,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """验证"""
    
    model.eval()
    criterion.eval()
    
    total_loss = 0
    loss_dict_accumulated = {}
    
    pbar = tqdm(dataloader, desc='Validating')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        gt_labels, gt_masks = prepare_instance_targets({'mask': masks})
        
        outputs = model(images)
        losses = criterion(outputs, gt_labels, gt_masks)
        
        loss = sum(losses.values())
        total_loss += loss.item()
        
        for k, v in losses.items():
            if k not in loss_dict_accumulated:
                loss_dict_accumulated[k] = 0
            loss_dict_accumulated[k] += v.item()
    
    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in loss_dict_accumulated.items()}
    avg_losses['total_loss'] = total_loss / num_batches
    
    return avg_losses


# ============================================================================
# 主训练循环
# ============================================================================

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # ===== 1. 创建DataLoader =====
    print("\n[1] Creating DataLoaders...")
    
    train_dataset = NPYSegmentationDataset(
        root_dir=args.data_root,
        split='train',
        image_size=args.image_size,
        transforms=get_default_transforms(args.image_size, 'train'),
        num_classes=args.num_classes,
    )
    
    val_dataset = NPYSegmentationDataset(
        root_dir=args.data_root,
        split='val',
        image_size=args.image_size,
        transforms=get_default_transforms(args.image_size, 'val'),
        num_classes=args.num_classes,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
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
    
    # ===== 2. 创建模型 =====
    print("\n[2] Creating model...")
    
    if args.model_size == 'tiny':
        model = segmamba_mask2former_tiny(
            num_classes=args.num_classes,
            num_queries=args.num_queries,
            in_chans=1,
        )
    elif args.model_size == 'small':
        model = segmamba_mask2former_small(
            num_classes=args.num_classes,
            num_queries=args.num_queries,
            in_chans=1,
        )
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {args.model_size}")
    print(f"  Parameters: {num_params / 1e6:.2f}M")
    
    # ===== 3. 创建Matcher和Loss =====
    print("\n[3] Creating loss function...")
    
    matcher = HungarianMatcher(
        cost_class=args.cost_class,
        cost_mask=args.cost_mask,
        cost_dice=args.cost_dice,
        num_points=args.num_points,
    )
    
    criterion = Mask2FormerLoss(
        num_classes=args.num_classes,
        matcher=matcher,
        weight_dict={
            'loss_ce': args.weight_ce,
            'loss_mask': args.weight_mask,
            'loss_dice': args.weight_dice,
        },
        eos_coef=args.eos_coef,
        num_points=args.num_points,
    )
    
    criterion = criterion.to(device)
    
    # ===== 4. 创建优化器 =====
    print("\n[4] Creating optimizer...")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    
    # ===== 5. 训练循环 =====
    print("\n[5] Starting training...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)
        
        # 训练
        train_loss_dict = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch
        )
        
        # 验证
        val_loss_dict = validate(model, criterion, val_loader, device)
        
        # 学习率调度
        scheduler.step()
        
        # 打印结果
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss_dict['total_loss']:.4f}")
        print(f"    - CE: {train_loss_dict['loss_ce']:.4f}")
        print(f"    - Mask: {train_loss_dict['loss_mask']:.4f}")
        print(f"    - Dice: {train_loss_dict['loss_dice']:.4f}")
        print(f"  Val Loss: {val_loss_dict['total_loss']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存loss历史
        train_losses.append(train_loss_dict)
        val_losses.append(val_loss_dict)
        
        # 保存checkpoint
        if epoch % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss_dict,
                'val_loss': val_loss_dict,
            }
            torch.save(
                checkpoint,
                output_dir / f'checkpoint_epoch_{epoch}.pth'
            )
            print(f"  ✓ Saved checkpoint: checkpoint_epoch_{epoch}.pth")
        
        # 保存最佳模型
        if val_loss_dict['total_loss'] < best_val_loss:
            best_val_loss = val_loss_dict['total_loss']
            torch.save(
                model.state_dict(),
                output_dir / 'best_model.pth'
            )
            print(f"  ✓ New best model! Val loss: {best_val_loss:.4f}")
    
    # 保存最终模型
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    
    # 保存loss历史
    np.save(output_dir / 'train_losses.npy', train_losses)
    np.save(output_dir / 'val_losses.npy', val_losses)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Output dir: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SegMamba + Mask2Former')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of NPY dataset')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    parser.add_argument('--num_classes', type=int, default=1,
                       help='Number of classes (excluding background)')
    
    # 模型参数
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['tiny', 'small'],
                       help='Model size')
    parser.add_argument('--num_queries', type=int, default=20,
                       help='Number of queries')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Loss权重（来自Mask2Former论文）
    parser.add_argument('--cost_class', type=float, default=2.0,
                       help='Classification cost weight for matching')
    parser.add_argument('--cost_mask', type=float, default=5.0,
                       help='Mask cost weight for matching')
    parser.add_argument('--cost_dice', type=float, default=5.0,
                       help='Dice cost weight for matching')
    parser.add_argument('--weight_ce', type=float, default=2.0,
                       help='Classification loss weight')
    parser.add_argument('--weight_mask', type=float, default=5.0,
                       help='Mask BCE loss weight')
    parser.add_argument('--weight_dice', type=float, default=5.0,
                       help='Dice loss weight')
    parser.add_argument('--eos_coef', type=float, default=0.1,
                       help='Background class weight')
    parser.add_argument('--num_points', type=int, default=12544,
                       help='Number of points for sampling')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    main(args)