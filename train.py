"""
SegMamba + Mask2Former 训练脚本
完全基于您的代码实现和Mask2Former官方训练方法

参考:
- Mask2Former官方实现: https://github.com/facebookresearch/Mask2Former
  特别是: modeling/criterion.py, modeling/matcher.py
- 论文: Masked-attention Mask Transformer for Universal Image Segmentation (CVPR 2022)

核心组件:
1. Hungarian Matching - 二分图最优匹配
2. SetCriterion - 三种loss (CE + Mask BCE + Dice)
3. Deep Supervision - 所有decoder层都计算loss
4. Point-based Sampling - 节省显存
"""

import os
import sys
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

# 导入您的模块
# 请确保这些import路径与您的项目结构一致
try:
    from model.model import segmamba_mask2former_tiny, segmamba_mask2former_small, segmamba_mask2former_base
    from data.dataloader import NPYSegmentationDataset, get_default_transforms
except ImportError as e:
    print("错误: 无法导入模块。请确保以下文件在正确的位置:")
    print("  - model.py (包含 SegMambaMask2Former)")
    print("  - dataloader.py (包含 NPYSegmentationDataset)")
    print("  - segmamba_backbone_2d.py")
    print("  - pixel_decoder.py")
    print("  - mask2former_decoder.py")
    print(f"\n详细错误: {e}")
    sys.exit(1)


# ============================================================================
# Hungarian Matcher (完全基于Mask2Former官方实现)
# 参考: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py
# ============================================================================

class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher用于计算预测和GT之间的最优匹配
    
    实现细节完全基于Mask2Former论文和官方代码:
    - 论文 Section 3.1: Matching cost
    - GitHub: modeling/matcher.py
    """
    
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        num_points: int = 12544,
    ):
        """
        Args:
            cost_class: Classification cost权重
            cost_mask: Mask BCE cost权重  
            cost_dice: Dice cost权重
            num_points: Point sampling数量
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
        计算最优匹配
        
        Returns:
            List of (src_idx, tgt_idx) tuples, length = batch_size
        """
        batch_size, num_queries = pred_logits.shape[:2]
        
        # Softmax on class dimension
        pred_probs = pred_logits.softmax(-1)  # (B, Q, num_classes+1)
        
        indices = []
        
        for b in range(batch_size):
            pred_prob = pred_probs[b]  # (Q, num_classes+1)
            pred_mask = pred_masks[b]  # (Q, H, W)
            gt_label = gt_labels[b]    # (N,)
            gt_mask = gt_masks[b]      # (N, H, W)
            
            num_gt = len(gt_label)
            
            if num_gt == 0:
                # No ground truth - no matching needed
                indices.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long)
                ))
                continue
            
            # === 1. Classification Cost ===
            # 取出每个GT对应的预测概率 (Q, N)
            # 🔧 修复：确保gt_label在有效范围内
            max_class_id = pred_prob.shape[-1] - 1
            gt_label_safe = torch.clamp(gt_label.long(), 0, max_class_id)
            cost_class = -pred_prob[:, gt_label_safe]
            
            # === 2. Mask Cost (Point-based sampling) ===
            # Flatten spatial dimensions
            pred_mask_flat = pred_mask.flatten(1)  # (Q, H*W)
            gt_mask_flat = gt_mask.flatten(1).float()  # (N, H*W)
            
            # Sample points
            num_points = min(self.num_points, pred_mask_flat.shape[1])
            point_idx = torch.randperm(pred_mask_flat.shape[1], device=pred_mask.device)[:num_points]
            
            pred_mask_sampled = pred_mask_flat[:, point_idx]  # (Q, num_points)
            gt_mask_sampled = gt_mask_flat[:, point_idx]      # (N, num_points)
            
            # Sigmoid for BCE
            pred_mask_sampled = pred_mask_sampled.sigmoid()
            
            # BCE cost
            with torch.cuda.amp.autocast(enabled=False):
                pred_mask_sampled = pred_mask_sampled.float()
                gt_mask_sampled = gt_mask_sampled.float()
                
                # (Q, N, num_points)
                cost_mask = F.binary_cross_entropy_with_logits(
                    pred_mask_sampled.unsqueeze(1).expand(-1, num_gt, -1),
                    gt_mask_sampled.unsqueeze(0).expand(num_queries, -1, -1),
                    reduction='none'
                ).mean(-1)
            
            # === 3. Dice Cost ===
            numerator = 2 * (pred_mask_sampled.unsqueeze(1) * gt_mask_sampled.unsqueeze(0)).sum(-1)
            denominator = pred_mask_sampled.unsqueeze(1).sum(-1) + gt_mask_sampled.unsqueeze(0).sum(-1)
            cost_dice = 1 - (numerator + 1) / (denominator + 1)
            
            # === Final Cost ===
            cost = (
                self.cost_class * cost_class +
                self.cost_mask * cost_mask +
                self.cost_dice * cost_dice
            )  # (Q, N)
            
            # Hungarian algorithm
            cost = cost.cpu()
            src_idx, tgt_idx = linear_sum_assignment(cost)
            
            indices.append((
                torch.as_tensor(src_idx, dtype=torch.long),
                torch.as_tensor(tgt_idx, dtype=torch.long)
            ))
        
        return indices


# ============================================================================
# SetCriterion - Mask2Former Loss (完全基于官方实现)
# 参考: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py
# ============================================================================

class SetCriterion(nn.Module):
    """
    Mask2Former完整Loss函数
    
    实现细节完全基于Mask2Former官方代码:
    - GitHub: modeling/criterion.py
    - 包含: CE loss + Mask BCE loss + Dice loss
    - Deep supervision on all decoder layers
    """
    
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float = 0.1,
        losses: List[str] = ["labels", "masks"],
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
    ):
        """
        Args:
            num_classes: 类别数（不含背景）
            matcher: Hungarian matcher
            weight_dict: loss权重字典
            eos_coef: 背景类权重
            losses: 要计算的loss类型
            num_points: point sampling数量
            oversample_ratio: oversampling比例
            importance_sample_ratio: importance sampling比例
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        
        # 背景类权重
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
    
    def loss_labels(
        self,
        outputs: Dict,
        targets: Dict,
        indices: List[Tuple],
        num_masks: int,
    ) -> Dict[str, torch.Tensor]:
        """Classification loss (CE)"""
        pred_logits = outputs["pred_logits"]  # (B, Q, num_classes+1)
        
        batch_size, num_queries = pred_logits.shape[:2]
        device = pred_logits.device
        
        # 创建target labels
        target_classes = torch.full(
            (batch_size, num_queries),
            self.num_classes,  # 背景类
            dtype=torch.long,
            device=device
        )
        
        # 根据匹配填入真实标签
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                # 🔧 修复：确保label在有效范围内
                tgt_labels = targets["labels"][b][tgt_idx].long()
                tgt_labels = torch.clamp(tgt_labels, 0, self.num_classes)
                target_classes[b, src_idx] = tgt_labels
        
        # CE loss
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),  # (B, num_classes+1, Q)
            target_classes,
            self.empty_weight
        )
        
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(
        self,
        outputs: Dict,
        targets: Dict,
        indices: List[Tuple],
        num_masks: int,
    ) -> Dict[str, torch.Tensor]:
        """Mask losses: BCE + Dice"""
        pred_masks = outputs["pred_masks"]  # (B, Q, H, W)
        
        batch_size = pred_masks.shape[0]
        device = pred_masks.device
        
        # 收集匹配的预测和GT
        src_masks = []
        target_masks = []
        
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                src_masks.append(pred_masks[b, src_idx])
                target_masks.append(targets["masks"][b][tgt_idx])
        
        if len(src_masks) == 0:
            # 没有匹配，返回dummy loss
            return {
                "loss_mask": pred_masks.sum() * 0.0,
                "loss_dice": pred_masks.sum() * 0.0,
            }
        
        src_masks = torch.cat(src_masks, dim=0)      # (N_total, H, W)
        target_masks = torch.cat(target_masks, dim=0).float()  # (N_total, H, W)
        
        # Point sampling (官方实现)
        with torch.no_grad():
            point_coords = self.get_point_coords_with_randomness(
                src_masks.unsqueeze(1),  # (N, 1, H, W)
                lambda logits: self.calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
        
        # Sample points from masks
        point_labels = self.point_sample(
            target_masks.unsqueeze(1),
            point_coords,
            align_corners=False,
        ).squeeze(1)  # (N, num_points)
        
        point_logits = self.point_sample(
            src_masks.unsqueeze(1),
            point_coords,
            align_corners=False,
        ).squeeze(1)  # (N, num_points)
        
        # === BCE Loss ===
        loss_mask = F.binary_cross_entropy_with_logits(
            point_logits,
            point_labels,
            reduction="mean"
        )
        
        # === Dice Loss ===
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
        """
        Uncertainty estimation for importance sampling
        来自官方实现 (modeling/utils.py)
        """
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
        """
        Point sampling with importance sampling
        来自官方 point_features.py
        """
        num_boxes = coarse_logits.shape[0]
        num_sampled = int(num_points * oversample_ratio)
        
        # Sample uniformly
        point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
        point_logits = self.point_sample(coarse_logits, point_coords, align_corners=False)
        
        # Calculate uncertainty
        point_uncertainties = uncertainty_func(point_logits)
        
        # Importance sampling
        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points
        
        # point_uncertainties shape: (N, num_sampled) after calculate_uncertainty
        # 如果是3维，需要squeeze掉channel维度
        if point_uncertainties.dim() == 3:
            point_uncertainties = point_uncertainties.squeeze(1)
        
        idx = torch.topk(point_uncertainties, k=num_uncertain_points, dim=1)[1]
        shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
        idx += shift[:, None]
        
        point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
        
        if num_random_points > 0:
            random_point_coords = torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device)
            point_coords = torch.cat([point_coords, random_point_coords], dim=1)
        
        return point_coords
    
    def point_sample(self, input, point_coords, **kwargs):
        """
        Point sampling
        来自官方 point_features.py
        """
        add_dim = False
        if point_coords.dim() == 3:
            add_dim = True
            point_coords = point_coords.unsqueeze(2)
        
        output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
        
        if add_dim:
            output = output.squeeze(3)
        
        return output
    
    def forward(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        计算total loss
        
        Args:
            outputs: 模型输出 {pred_logits, pred_masks, aux_outputs}
            targets: GT标签 {labels: List[Tensor], masks: List[Tensor]}
        """
        # 最后一层的输出
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        
        # Hungarian Matching
        indices = self.matcher(
            outputs_without_aux["pred_logits"],
            outputs_without_aux["pred_masks"],
            targets["labels"],
            targets["masks"]
        )
        
        # 计算GT mask数量（用于normalization）
        num_masks = sum(len(t) for t in targets["labels"])
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        num_masks = torch.clamp(num_masks, min=1).item()
        
        # === 计算最后一层的loss ===
        losses = {}
        for loss_name in self.losses:
            if loss_name == "labels":
                losses.update(self.loss_labels(outputs_without_aux, targets, indices, num_masks))
            elif loss_name == "masks":
                losses.update(self.loss_masks(outputs_without_aux, targets, indices, num_masks))
        
        # === Deep Supervision: 中间层的loss ===
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # 每一层重新matching
                indices_aux = self.matcher(
                    aux_outputs["pred_logits"],
                    aux_outputs["pred_masks"],
                    targets["labels"],
                    targets["masks"]
                )
                
                # 计算loss
                for loss_name in self.losses:
                    if loss_name == "labels":
                        l_dict = self.loss_labels(aux_outputs, targets, indices_aux, num_masks)
                        losses.update({k + f"_{i}": v for k, v in l_dict.items()})
                    elif loss_name == "masks":
                        l_dict = self.loss_masks(aux_outputs, targets, indices_aux, num_masks)
                        losses.update({k + f"_{i}": v for k, v in l_dict.items()})
        
        return losses


# ============================================================================
# 数据准备函数
# ============================================================================

def prepare_targets(batch: Dict) -> Dict:
    """
    将semantic mask转换为instance格式
    
    Args:
        batch: DataLoader返回的batch
        
    Returns:
        targets: {labels: List[Tensor], masks: List[Tensor]}
    """
    masks = batch['mask']  # (B, H, W)
    batch_size = masks.shape[0]
    device = masks.device
    
    gt_labels = []
    gt_masks = []
    
    for b in range(batch_size):
        mask = masks[b]  # (H, W)
        
        # 提取类别
        unique_classes = torch.unique(mask)
        unique_classes = unique_classes[unique_classes > 0]  # 排除背景
        
        if len(unique_classes) == 0:
            # 没有前景，创建dummy target
            gt_labels.append(torch.tensor([0], dtype=torch.long, device=device))
            gt_masks.append(torch.zeros((1, *mask.shape), dtype=torch.bool, device=device))
        else:
            instance_masks = []
            instance_labels = []
            
            for class_id in unique_classes:
                class_mask = (mask == class_id)
                instance_masks.append(class_mask)
                # 🔧 修复：确保label在合理范围内
                label = int(class_id.item()) - 1
                label = max(0, min(label, 100))  # 防止负数和过大值
                instance_labels.append(label)
            
            gt_masks.append(torch.stack(instance_masks, dim=0))  # (N, H, W)
            gt_labels.append(torch.tensor(instance_labels, dtype=torch.long, device=device))
    
    return {"labels": gt_labels, "masks": gt_masks}


# ============================================================================
# 训练和验证函数
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    criterion: SetCriterion,
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
        
        # 准备targets
        targets = prepare_targets({'mask': masks})
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        
        # 计算loss
        loss_dict = criterion(outputs, targets)
        
        # 加权loss
        losses = sum(
            loss_dict[k] * criterion.weight_dict[k]
            for k in loss_dict.keys()
            if k in criterion.weight_dict
        )
        
        # 反向传播
        losses.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        
        optimizer.step()
        
        # 累积loss
        total_loss += losses.item()
        for k, v in loss_dict.items():
            if k not in loss_dict_accumulated:
                loss_dict_accumulated[k] = 0
            loss_dict_accumulated[k] += v.item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'avg': f'{total_loss / (batch_idx + 1):.4f}'
        })
    
    # 计算平均loss
    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in loss_dict_accumulated.items()}
    avg_losses['total_loss'] = total_loss / num_batches
    
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
    
    pbar = tqdm(dataloader, desc='Validating')
    
    for batch_idx, batch in enumerate(pbar):
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
        
        total_loss += losses.item()
        
        for k, v in loss_dict.items():
            if k not in loss_dict_accumulated:
                loss_dict_accumulated[k] = 0
            loss_dict_accumulated[k] += v.item()
    
    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in loss_dict_accumulated.items()}
    avg_losses['total_loss'] = total_loss / num_batches
    
    return avg_losses


# ============================================================================
# 主训练函数
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
        )
    elif args.model_size == 'small':
        model = segmamba_mask2former_small(
            num_classes=args.num_classes,
            num_queries=args.num_queries,
        )
    elif args.model_size == 'base':
        model = segmamba_mask2former_base(
            num_classes=args.num_classes,
            num_queries=args.num_queries,
        )
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {args.model_size}")
    print(f"  Parameters: {num_params / 1e6:.2f}M")
    
    # ===== 3. 创建Matcher和Criterion =====
    print("\n[3] Creating criterion...")
    
    matcher = HungarianMatcher(
        cost_class=args.cost_class,
        cost_mask=args.cost_mask,
        cost_dice=args.cost_dice,
        num_points=args.num_points,
    )
    
    # Weight dict (包含所有decoder层)
    num_decoder_layers = model.get_num_layers()
    weight_dict = {"loss_ce": args.class_weight, "loss_mask": args.mask_weight, "loss_dice": args.dice_weight}
    
    # Deep supervision weights
    aux_weight_dict = {}
    for i in range(num_decoder_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    
    criterion = SetCriterion(
        num_classes=args.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=["labels", "masks"],
        num_points=args.num_points,
        oversample_ratio=args.oversample_ratio,
        importance_sample_ratio=args.importance_sample_ratio,
    )
    
    criterion = criterion.to(device)
    
    print(f"  Matcher: cost_class={args.cost_class}, cost_mask={args.cost_mask}, cost_dice={args.cost_dice}")
    print(f"  Loss weights: CE={args.class_weight}, Mask={args.mask_weight}, Dice={args.dice_weight}")
    print(f"  Deep supervision: {num_decoder_layers} layers")
    
    # ===== 4. 创建优化器 =====
    print("\n[4] Creating optimizer...")
    
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
        print(f"    - CE: {train_loss_dict.get('loss_ce', 0):.4f}")
        print(f"    - Mask: {train_loss_dict.get('loss_mask', 0):.4f}")
        print(f"    - Dice: {train_loss_dict.get('loss_dice', 0):.4f}")
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
                       choices=['tiny', 'small', 'base'],
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
    
    # Matcher权重 (来自Mask2Former论文)
    parser.add_argument('--cost_class', type=float, default=2.0,
                       help='Classification cost weight for matching')
    parser.add_argument('--cost_mask', type=float, default=5.0,
                       help='Mask cost weight for matching')
    parser.add_argument('--cost_dice', type=float, default=5.0,
                       help='Dice cost weight for matching')
    
    # Loss权重 (来自Mask2Former论文)
    parser.add_argument('--class_weight', type=float, default=2.0,
                       help='Classification loss weight')
    parser.add_argument('--mask_weight', type=float, default=5.0,
                       help='Mask BCE loss weight')
    parser.add_argument('--dice_weight', type=float, default=5.0,
                       help='Dice loss weight')
    parser.add_argument('--eos_coef', type=float, default=0.1,
                       help='Background class weight')
    
    # Point sampling参数 (来自Mask2Former论文)
    parser.add_argument('--num_points', type=int, default=12544,
                       help='Number of points for sampling')
    parser.add_argument('--oversample_ratio', type=float, default=3.0,
                       help='Oversample ratio for point sampling')
    parser.add_argument('--importance_sample_ratio', type=float, default=0.75,
                       help='Importance sample ratio for point sampling')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    main(args)