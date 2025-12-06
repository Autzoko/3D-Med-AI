"""
SegMamba 2D Simple - 训练脚本
使用简单的上采样decoder，不用Mask2Former
预期Dice: 0.85-0.90+
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json

# 导入模型和数据
try:
    from model.model_simple import segmamba2d_simple_tiny, segmamba2d_simple_small, segmamba2d_simple_base
    from data.dataloader import NPYSegmentationDataset, get_default_transforms
except ImportError as e:
    print(f"Error: {e}")
    print("Please ensure segmamba2d_simple.py and dataloader.py are available")
    exit(1)


# ============================================================================
# Loss Functions
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) logits
            target: (B, 1, H, W) ground truth (0 or 1)
        """
        pred = pred.sigmoid()
        
        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """BCE + Dice Loss"""
    def __init__(self, bce_weight=1.0, dice_weight=2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        loss_bce = self.bce(pred, target)
        loss_dice = self.dice(pred, target)
        
        total_loss = self.bce_weight * loss_bce + self.dice_weight * loss_dice
        
        return {
            'total': total_loss,
            'bce': loss_bce,
            'dice': loss_dice,
        }


# ============================================================================
# Metrics
# ============================================================================

@torch.no_grad()
def calculate_dice_score(pred, target, threshold=0.5):
    """
    计算Dice Score
    
    Args:
        pred: (B, 1, H, W) logits
        target: (B, 1, H, W) ground truth
    
    Returns:
        dice_score: float
    """
    pred_binary = (pred.sigmoid() > threshold).float()
    
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    
    return dice.item()


# ============================================================================
# Training and Validation
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    total_bce = 0
    total_dice_loss = 0
    total_dice_score = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch in pbar:
        images = batch['image'].to(device)  # (B, 1, H, W)
        masks = batch['mask'].to(device)    # (B, H, W)
        
        # 确保masks是(B, 1, H, W)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        masks = masks.float()
        
        # 前向传播
        optimizer.zero_grad()
        pred = model(images)  # (B, 1, H, W)
        
        # 计算loss
        losses = criterion(pred, masks)
        loss = losses['total']
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算Dice Score
        with torch.no_grad():
            dice_score = calculate_dice_score(pred, masks)
        
        # 累积
        total_loss += loss.item()
        total_bce += losses['bce'].item()
        total_dice_loss += losses['dice'].item()
        total_dice_score += dice_score
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice_score:.4f}'
        })
    
    return {
        'total_loss': total_loss / num_batches,
        'bce_loss': total_bce / num_batches,
        'dice_loss': total_dice_loss / num_batches,
        'dice_score': total_dice_score / num_batches,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    
    total_loss = 0
    total_bce = 0
    total_dice_loss = 0
    total_dice_score = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Validating')
    
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        masks = masks.float()
        
        # 前向传播
        pred = model(images)
        
        # 计算loss
        losses = criterion(pred, masks)
        
        # 计算Dice Score
        dice_score = calculate_dice_score(pred, masks)
        
        # 累积
        total_loss += losses['total'].item()
        total_bce += losses['bce'].item()
        total_dice_loss += losses['dice'].item()
        total_dice_score += dice_score
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{losses["total"].item():.4f}',
            'dice': f'{dice_score:.4f}'
        })
    
    return {
        'total_loss': total_loss / num_batches,
        'bce_loss': total_bce / num_batches,
        'dice_loss': total_dice_loss / num_batches,
        'dice_score': total_dice_score / num_batches,
    }


# ============================================================================
# Main
# ============================================================================

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # ===== 1. 数据 =====
    print("[1] Loading data...")
    
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
    print(f"  Val: {len(val_dataset)} samples\n")
    
    # ===== 2. 模型 =====
    print("[2] Creating model...")
    
    if args.model_size == 'tiny':
        model = segmamba2d_simple_tiny(
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path_rate,
        )
    elif args.model_size == 'small':
        model = segmamba2d_simple_small(
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path_rate,
        )
    elif args.model_size == 'base':
        model = segmamba2d_simple_base(
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path_rate,
        )
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    model = model.to(device)
    
    info = model.get_model_info()
    print(f"  Parameters: {info['num_parameters'] / 1e6:.2f}M\n")
    
    # ===== 3. Loss和Optimizer =====
    print("[3] Creating loss and optimizer...")
    
    criterion = CombinedLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    
    print(f"  BCE weight: {args.bce_weight}")
    print(f"  Dice weight: {args.dice_weight}\n")
    
    # ===== 4. 训练 =====
    print("[4] Starting training...")
    print("=" * 80)
    
    best_dice = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)
        
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device)
        
        # 学习率调度
        scheduler.step()
        
        # 打印结果
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train - Loss: {train_metrics['total_loss']:.4f}, Dice: {train_metrics['dice_score']:.4f}")
        print(f"  Val   - Loss: {val_metrics['total_loss']:.4f}, Dice: {val_metrics['dice_score']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_metrics['dice_score'] > best_dice:
            best_dice = val_metrics['dice_score']
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print(f"  ✓ New best model! Dice: {best_dice:.4f}")
        
        # 保存checkpoint
        if epoch % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pth')
            print(f"  ✓ Saved checkpoint")
    
    # 保存最终模型
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    
    print("\n" + "=" * 80)
    print(f"Training completed! Best Dice: {best_dice:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SegMamba 2D Simple Training')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to processed dataset')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--num_classes', type=int, default=1,
                       help='Number of classes')
    
    # 模型参数
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['tiny', 'small', 'base'],
                       help='Model size')
    parser.add_argument('--drop_path_rate', type=float, default=0.0,
                       help='DropPath rate')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers')
    
    # Loss权重
    parser.add_argument('--bce_weight', type=float, default=1.0,
                       help='BCE loss weight')
    parser.add_argument('--dice_weight', type=float, default=2.0,
                       help='Dice loss weight')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='./outputs_simple',
                       help='Output directory')
    parser.add_argument('--save_every', type=int, default=20,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    main(args)