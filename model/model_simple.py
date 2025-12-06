"""
SegMamba 2D - Simple Version
使用SegMamba backbone + 简单decoder
不用Mask2Former，直接像素级预测
"""

import torch
import torch.nn as nn
from typing import Dict
import sys
sys.path.append('.')

try:
    from model.backbone.segmamba_backbone_2d import SegMambaBackbone2D
    from model.detector.simple_decoder import SimpleSegDecoder
except ImportError:
    print("Warning: Using relative import")
    from backbone.segmamba_backbone_2d import SegMambaBackbone2D


class SegMamba2DSimple(nn.Module):
    """
    SegMamba 2D Simple版本
    
    架构：
        Input (B, 1, H, W)
        → SegMamba Backbone → [f1, f2, f3, f4]
        → Simple FPN Decoder
        → Output (B, num_classes, H, W)
    
    特点：
        • 简单直接
        • 全像素监督
        • 类似SegMamba原版
        • 预期Dice: 0.85-0.90+
    """
    
    def __init__(
        self,
        backbone_name='small',
        in_chans=1,
        num_classes=1,
        decoder_dim=256,
        drop_path_rate=0.0,
        dropout=0.1,
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        # Backbone维度映射
        backbone_dims = {
            'tiny': [48, 96, 192, 384],
            'small': [48, 96, 192, 384],
            'base': [64, 128, 256, 512],
        }
        
        if backbone_name not in backbone_dims:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        encoder_dims = backbone_dims[backbone_name]
        
        print(f"[SegMamba2DSimple] Creating {backbone_name} model...")
        
        # 1. Backbone
        self.backbone = SegMambaBackbone2D(
            model_name=backbone_name,
            in_chans=in_chans,
            drop_path_rate=drop_path_rate,
        )
        print(f"  ✓ Backbone: dims={encoder_dims}")
        
        # 2. Simple Decoder
        self.decoder = SimpleSegDecoder(
            encoder_dims=encoder_dims,
            decoder_dim=decoder_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        print(f"  ✓ Decoder: dim={decoder_dim}, classes={num_classes}")
        
        print("[SegMamba2DSimple] Model created successfully!")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, C, H, W) 输入图像
        
        Returns:
            out: (B, num_classes, H, W) 分割预测（logits）
        """
        # Backbone
        features = self.backbone(x)  # [f1, f2, f3, f4]
        
        # Decoder
        out = self.decoder(features)  # (B, num_classes, H, W)
        
        return out
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        num_params = sum(p.numel() for p in self.parameters())
        return {
            'backbone': self.backbone_name,
            'num_classes': self.num_classes,
            'num_parameters': num_params,
            'decoder_type': 'simple_fpn',
        }


# ============================================================================
# Convenience functions
# ============================================================================

def segmamba2d_simple_tiny(num_classes=1, drop_path_rate=0.0):
    """Tiny version"""
    return SegMamba2DSimple(
        backbone_name='tiny',
        in_chans=1,
        num_classes=num_classes,
        decoder_dim=256,
        drop_path_rate=drop_path_rate,
    )


def segmamba2d_simple_small(num_classes=1, drop_path_rate=0.0):
    """Small version (推荐)"""
    return SegMamba2DSimple(
        backbone_name='small',
        in_chans=1,
        num_classes=num_classes,
        decoder_dim=256,
        drop_path_rate=drop_path_rate,
    )


def segmamba2d_simple_base(num_classes=1, drop_path_rate=0.0):
    """Base version"""
    return SegMamba2DSimple(
        backbone_name='base',
        in_chans=1,
        num_classes=num_classes,
        decoder_dim=256,
        drop_path_rate=drop_path_rate,
    )


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    print("Testing SegMamba2DSimple...")
    print("=" * 80)
    
    # 创建模型
    model = segmamba2d_simple_small(num_classes=1)
    
    # 打印信息
    info = model.get_model_info()
    print(f"\nModel Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    # 测试前向传播
    print(f"\nTesting forward pass...")
    B, C, H, W = 2, 1, 256, 256
    x = torch.randn(B, C, H, W)
    
    with torch.no_grad():
        out = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    
    assert out.shape == (B, 1, H, W), f"Expected (2, 1, 256, 256), got {out.shape}"
    
    # 测试loss计算
    print(f"\nTesting loss calculation...")
    gt_mask = torch.randint(0, 2, (B, 1, H, W)).float()
    
    # BCE Loss
    loss_bce = nn.functional.binary_cross_entropy_with_logits(out, gt_mask)
    
    # Dice Loss
    pred_sigmoid = out.sigmoid()
    intersection = (pred_sigmoid * gt_mask).sum()
    union = pred_sigmoid.sum() + gt_mask.sum()
    dice = (2 * intersection + 1) / (union + 1)
    loss_dice = 1 - dice
    
    total_loss = loss_bce + 2.0 * loss_dice
    
    print(f"  BCE Loss: {loss_bce.item():.4f}")
    print(f"  Dice Loss: {loss_dice.item():.4f}")
    print(f"  Total Loss: {total_loss.item():.4f}")
    
    # 反向传播测试
    print(f"\nTesting backward pass...")
    total_loss.backward()
    print(f"  ✓ Gradients computed successfully")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)