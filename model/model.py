"""
Complete SegMamba + Mask2Former Model
整合了backbone, pixel decoder, 和 transformer decoder的完整模型

Enhanced with:
- Encoder-decoder skip connections (U-Net style)
- DropPath regularization

Author: Based on Mask2Former (CVPR 2022) and SegMamba (MICCAI 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

# 导入我们的组件
from model.backbone.segmamba_backbone_2d import (
    SegMambaBackbone2D,
    segmamba_backbone_tiny,
    segmamba_backbone_small,
    segmamba_backbone_base,
)
from model.detector.pixel_decoder import SimpleFPNDecoder, BasePixelDecoder
from model.detector.mask2former_decoder import Mask2FormerTransformerDecoder


class SegMambaMask2Former(nn.Module):
    """
    完整的SegMamba + Mask2Former模型
    
    Enhanced with:
    - Encoder-decoder skip connections (use_skip_connections parameter)
    - DropPath regularization (drop_path_rate parameter)
    """
    
    def __init__(
        self,
        # Backbone配置
        backbone_name: str = 'small',
        in_chans: int = 1,
        drop_path_rate: float = 0.0,  # 新增：DropPath正则化
        # Pixel decoder配置
        pixel_decoder_name: str = 'simple_fpn',
        conv_dim: int = 256,
        mask_dim: int = 256,
        use_skip_connections: bool = False,  # 新增：Encoder-Decoder skip connections
        # Transformer decoder配置
        hidden_dim: int = 256,
        num_queries: int = 30,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dec_layers: int = 6,
        pre_norm: bool = False,
        # 任务配置
        num_classes: int = 2,
        enforce_input_project: bool = False,
    ):
        """
        新增参数:
            drop_path_rate: DropPath rate for backbone (0.0 = disabled, 0.1-0.15 recommended)
            use_skip_connections: Enable encoder-decoder skip connections (U-Net style)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.backbone_name = backbone_name
        
        # ==================== 1. 创建Backbone ====================
        print(f"[Model] Creating {backbone_name} backbone...")
        
        if backbone_name == 'tiny':
            self.backbone = segmamba_backbone_tiny(
                in_chans=in_chans,
                return_features='all',
                drop_path_rate=drop_path_rate  # 新增参数
            )
            backbone_dims = [48, 96, 192, 384]
        elif backbone_name == 'small':
            self.backbone = segmamba_backbone_small(
                in_chans=in_chans,
                return_features='all',
                drop_path_rate=drop_path_rate  # 新增参数
            )
            backbone_dims = [48, 96, 192, 384]
        elif backbone_name == 'base':
            self.backbone = segmamba_backbone_base(
                in_chans=in_chans,
                return_features='all',
                drop_path_rate=drop_path_rate  # 新增参数
            )
            backbone_dims = [64, 128, 256, 512]
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        print(f"  ✓ Backbone dims: {backbone_dims}")
        if drop_path_rate > 0:
            print(f"  ✓ DropPath rate: {drop_path_rate}")
        
        # ==================== 2. 创建Pixel Decoder ====================
        print(f"[Model] Creating {pixel_decoder_name} pixel decoder...")
        
        if pixel_decoder_name == 'simple_fpn':
            self.pixel_decoder = SimpleFPNDecoder(
                in_channels_list=backbone_dims,
                conv_dim=conv_dim,
                mask_dim=mask_dim,
                use_skip_connections=use_skip_connections  # 新增参数
            )
        elif pixel_decoder_name == 'base_fpn':
            self.pixel_decoder = BasePixelDecoder(
                in_channels_list=backbone_dims,
                conv_dim=conv_dim,
                mask_dim=mask_dim,
            )
        else:
            raise ValueError(f"Unknown pixel decoder: {pixel_decoder_name}")
        
        print(f"  ✓ Pixel decoder: conv_dim={conv_dim}, mask_dim={mask_dim}")
        if use_skip_connections:
            print(f"  ✓ Encoder-decoder skip connections enabled")
        
        # ==================== 3. 创建Mask2Former Decoder ====================
        print(f"[Model] Creating Mask2Former transformer decoder...")
        
        self.transformer_decoder = Mask2FormerTransformerDecoder(
            in_channels=conv_dim,
            mask_classification=True,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_project,
        )
        
        print(f"  ✓ Transformer decoder: {num_queries} queries, {dec_layers} layers")
        print(f"[Model] Model created successfully!")
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, C, H, W)
            
        Returns:
            字典包含:
                - pred_logits: (B, Q, num_classes+1) 类别预测
                - pred_masks: (B, Q, H/4, W/4) mask预测
                - aux_outputs: list of dicts，中间层预测
        """
        # 1. Backbone: 提取多尺度特征
        backbone_features = self.backbone(x)
        
        # 2. Pixel Decoder: 生成mask features和multi-scale features
        mask_features, multi_scale_features = self.pixel_decoder(backbone_features)
        
        # 3. Transformer Decoder: Query-based预测
        predictions = self.transformer_decoder(multi_scale_features, mask_features)
        
        return predictions
    
    def get_num_layers(self):
        """获取decoder层数（用于deep supervision）"""
        return self.transformer_decoder.num_layers + 1
    
    def get_model_info(self) -> Dict:
        """获取模型配置信息"""
        return {
            'backbone': self.backbone_name,
            'num_classes': self.num_classes,
            'num_queries': self.num_queries,
            'num_decoder_layers': self.transformer_decoder.num_layers,
            'hidden_dim': self.transformer_decoder.decoder_norm.normalized_shape[0],
        }


# ============================================================================
# Convenience functions (保持原有名称，添加新参数)
# ============================================================================

def segmamba_mask2former_tiny(
    num_classes=2, 
    num_queries=30,
    drop_path_rate=0.0,
    use_skip_connections=False
):
    """
    Tiny SegMamba + Mask2Former
    
    新增参数:
        drop_path_rate: DropPath正则化 (推荐: 0.05-0.1)
        use_skip_connections: Encoder-decoder skip connections
    """
    return SegMambaMask2Former(
        backbone_name='tiny',
        in_chans=1,
        num_classes=num_classes,
        num_queries=num_queries,
        drop_path_rate=drop_path_rate,
        use_skip_connections=use_skip_connections,
    )


def segmamba_mask2former_small(
    num_classes=2, 
    num_queries=30,
    drop_path_rate=0.0,
    use_skip_connections=False
):
    """
    Small SegMamba + Mask2Former
    
    新增参数:
        drop_path_rate: DropPath正则化 (推荐: 0.1-0.15)
        use_skip_connections: Encoder-decoder skip connections
    """
    return SegMambaMask2Former(
        backbone_name='small',
        in_chans=1,
        num_classes=num_classes,
        num_queries=num_queries,
        drop_path_rate=drop_path_rate,
        use_skip_connections=use_skip_connections,
    )


def segmamba_mask2former_base(
    num_classes=2, 
    num_queries=30,
    drop_path_rate=0.0,
    use_skip_connections=False
):
    """
    Base SegMamba + Mask2Former
    
    新增参数:
        drop_path_rate: DropPath正则化 (推荐: 0.15-0.2)
        use_skip_connections: Encoder-decoder skip connections
    """
    return SegMambaMask2Former(
        backbone_name='base',
        in_chans=1,
        num_classes=num_classes,
        num_queries=num_queries,
        drop_path_rate=drop_path_rate,
        use_skip_connections=use_skip_connections,
    )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Testing Enhanced SegMamba + Mask2Former")
    print("="*80)
    
    # Test 1: 原有API（向后兼容）
    print("\n[Test 1] 原有API（不使用新功能）")
    model_old = segmamba_mask2former_small(num_classes=1, num_queries=20)
    
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        outputs_old = model_old(x)
    
    print(f"Input: {x.shape}")
    print(f"pred_logits: {outputs_old['pred_logits'].shape}")
    print(f"pred_masks: {outputs_old['pred_masks'].shape}")
    print(f"aux_outputs: {len(outputs_old['aux_outputs'])} layers")
    
    # Test 2: 使用新功能
    print("\n[Test 2] 使用新功能（DropPath + Skip Connections）")
    model_new = segmamba_mask2former_small(
        num_classes=1,
        num_queries=20,
        drop_path_rate=0.1,  # DropPath
        use_skip_connections=True  # Skip connections
    )
    
    with torch.no_grad():
        outputs_new = model_new(x)
    
    print(f"pred_logits: {outputs_new['pred_logits'].shape}")
    print(f"pred_masks: {outputs_new['pred_masks'].shape}")
    
    # Count parameters
    params_old = sum(p.numel() for p in model_old.parameters())
    params_new = sum(p.numel() for p in model_new.parameters())
    
    print(f"\nParameters:")
    print(f"  Original: {params_old:,}")
    print(f"  Enhanced: {params_new:,}")
    print(f"  Increase: {params_new - params_old:,} (+{(params_new/params_old - 1)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("✓ Backward compatible!")
    print("="*80)