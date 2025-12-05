"""
Complete SegMamba + Mask2Former Model
整合了backbone, pixel decoder, 和 transformer decoder的完整模型

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
    
    架构流程:
        输入图像 (B, C, H, W)
            ↓
        SegMamba Backbone → 提取多尺度特征
            ↓
        Pixel Decoder (FPN) → 生成mask features和multi-scale features
            ↓
        Mask2Former Decoder → Query-based预测
            ↓
        输出: {pred_logits, pred_masks, aux_outputs}
    
    适用场景:
        - 医学超声图像分割
        - 域适应任务
        - 少样本学习
        - 实例/语义分割
    """
    
    def __init__(
        self,
        # Backbone配置
        backbone_name: str = 'small',  # 'tiny', 'small', 'base'
        in_chans: int = 1,
        # Pixel decoder配置
        pixel_decoder_name: str = 'simple_fpn',  # 'simple_fpn' or 'base_fpn'
        conv_dim: int = 256,
        mask_dim: int = 256,
        # Transformer decoder配置
        hidden_dim: int = 256,
        num_queries: int = 30,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dec_layers: int = 6,
        pre_norm: bool = False,
        # 任务配置
        num_classes: int = 2,  # 分类数（不包括背景）
        enforce_input_project: bool = False,
    ):
        """
        Args:
            backbone_name: SegMamba backbone大小 ('tiny', 'small', 'base')
            in_chans: 输入通道数 (1=灰度, 3=RGB)
            pixel_decoder_name: pixel decoder类型
            conv_dim: pixel decoder中间通道数
            mask_dim: mask特征维度
            hidden_dim: transformer隐藏层维度
            num_queries: object queries数量 (推荐: 20-30 for medical imaging)
            nheads: attention heads数量
            dim_feedforward: FFN维度
            dec_layers: decoder层数
            pre_norm: 是否使用pre-normalization
            num_classes: 分割类别数（不包括背景类）
            enforce_input_project: 是否强制使用input projection
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
                return_features='all'
            )
            backbone_dims = [48, 96, 192, 384]
        elif backbone_name == 'small':
            self.backbone = segmamba_backbone_small(
                in_chans=in_chans,
                return_features='all'
            )
            backbone_dims = [48, 96, 192, 384]
        elif backbone_name == 'base':
            self.backbone = segmamba_backbone_base(
                in_chans=in_chans,
                return_features='all'
            )
            backbone_dims = [64, 128, 256, 512]
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        print(f"  ✓ Backbone dims: {backbone_dims}")
        
        # ==================== 2. 创建Pixel Decoder ====================
        print(f"[Model] Creating {pixel_decoder_name} pixel decoder...")
        
        if pixel_decoder_name == 'simple_fpn':
            self.pixel_decoder = SimpleFPNDecoder(
                in_channels_list=backbone_dims,
                conv_dim=conv_dim,
                mask_dim=mask_dim,
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
                - pred_logits: (B, Q, num_classes+1) 类别预测（+1是背景类）
                - pred_masks: (B, Q, H/4, W/4) mask预测
                - aux_outputs: list of dicts，中间层预测（用于deep supervision）
                
        Example:
            >>> model = SegMambaMask2Former(num_classes=2)
            >>> images = torch.randn(4, 1, 256, 256)
            >>> predictions = model(images)
            >>> logits = predictions['pred_logits']  # (4, 30, 3)
            >>> masks = predictions['pred_masks']    # (4, 30, 64, 64)
        """
        # 1. Backbone: 提取多尺度特征
        # Output: [(B,48,H/2,W/2), (B,96,H/4,W/4), (B,192,H/8,W/8), (B,384,H/16,W/16)]
        backbone_features = self.backbone(x)
        
        # 2. Pixel Decoder: 生成mask features和multi-scale features
        # mask_features: (B, mask_dim, H/4, W/4)
        # multi_scale_features: [(B,C,H/32,W/32), (B,C,H/16,W/16), (B,C,H/8,W/8)]
        mask_features, multi_scale_features = self.pixel_decoder(backbone_features)
        
        # 3. Transformer Decoder: Query-based预测
        predictions = self.transformer_decoder(multi_scale_features, mask_features)
        
        return predictions
    
    def get_num_layers(self):
        """获取decoder层数（用于deep supervision）"""
        return self.transformer_decoder.num_layers + 1  # +1 for learnable queries
    
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
# 便捷函数：创建不同规模的模型
# ============================================================================

def segmamba_mask2former_tiny(
    num_classes: int = 2,
    num_queries: int = 20,
    in_chans: int = 1,
    **kwargs
) -> SegMambaMask2Former:
    """
    Tiny模型配置
    
    适用场景:
        - 快速实验
        - GPU显存受限
        - 实时推理
    
    规格:
        - Backbone: SegMamba Tiny (~5M params)
        - Queries: 20
        - Decoder layers: 3
        - 总参数: ~13M
    """
    return SegMambaMask2Former(
        backbone_name='tiny',
        in_chans=in_chans,
        num_classes=num_classes,
        num_queries=num_queries,
        hidden_dim=256,
        dec_layers=3,
        **kwargs
    )


def segmamba_mask2former_small(
    num_classes: int = 2,
    num_queries: int = 30,
    in_chans: int = 1,
    **kwargs
) -> SegMambaMask2Former:
    """
    Small模型配置（推荐用于医学图像）
    
    适用场景:
        - 医学图像分割
        - 域适应任务
        - 平衡性能和效率
    
    规格:
        - Backbone: SegMamba Small (~8M params)
        - Queries: 30
        - Decoder layers: 6
        - 总参数: ~18M
    """
    return SegMambaMask2Former(
        backbone_name='small',
        in_chans=in_chans,
        num_classes=num_classes,
        num_queries=num_queries,
        hidden_dim=256,
        dec_layers=6,
        **kwargs
    )


def segmamba_mask2former_base(
    num_classes: int = 2,
    num_queries: int = 50,
    in_chans: int = 1,
    **kwargs
) -> SegMambaMask2Former:
    """
    Base模型配置
    
    适用场景:
        - 最高性能需求
        - 大数据集
        - 复杂分割任务
    
    规格:
        - Backbone: SegMamba Base (~15M params)
        - Queries: 50
        - Decoder layers: 6
        - 总参数: ~30M
    """
    return SegMambaMask2Former(
        backbone_name='base',
        in_chans=in_chans,
        num_classes=num_classes,
        num_queries=num_queries,
        hidden_dim=256,
        dec_layers=6,
        **kwargs
    )


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SegMamba + Mask2Former 完整模型测试")
    print("="*80 + "\n")
    
    # ==================== 测试1: 创建模型 ====================
    print("[测试 1] 创建模型")
    print("-" * 40)
    model = segmamba_mask2former_small(num_classes=2, num_queries=30)
    print(f"✓ 模型创建成功")
    print(f"  模型信息: {model.get_model_info()}\n")
    
    # ==================== 测试2: 前向传播 ====================
    print("[测试 2] 前向传播")
    print("-" * 40)
    # 模拟输入: 4张256x256的灰度超声图像
    images = torch.randn(4, 1, 256, 256)
    print(f"输入形状: {images.shape}")
    
    with torch.no_grad():
        predictions = model(images)
    
    print(f"输出:")
    print(f"  pred_logits: {predictions['pred_logits'].shape}")
    print(f"  pred_masks: {predictions['pred_masks'].shape}")
    print(f"  aux_outputs: {len(predictions['aux_outputs'])} 个中间层预测")
    print(f"✓ 前向传播成功\n")
    
    # ==================== 测试3: 参数统计 ====================
    print("[测试 3] 参数统计")
    print("-" * 40)
    
    models = {
        'Tiny': segmamba_mask2former_tiny(),
        'Small': segmamba_mask2former_small(),
        'Base': segmamba_mask2former_base(),
    }
    
    for name, m in models.items():
        total_params = sum(p.numel() for p in m.parameters())
        backbone_params = sum(p.numel() for p in m.backbone.parameters())
        pixel_params = sum(p.numel() for p in m.pixel_decoder.parameters())
        decoder_params = sum(p.numel() for p in m.transformer_decoder.parameters())
        
        print(f"{name:10s}:")
        print(f"  总参数: {total_params/1e6:>6.2f}M")
        print(f"  └─ Backbone:  {backbone_params/1e6:>6.2f}M")
        print(f"  └─ Pixel Dec: {pixel_params/1e6:>6.2f}M")
        print(f"  └─ Trans Dec: {decoder_params/1e6:>6.2f}M")
    
    print()
    
    # ==================== 测试4: 不同输入尺寸 ====================
    print("[测试 4] 不同输入尺寸")
    print("-" * 40)
    
    model = segmamba_mask2former_small(num_classes=2)
    
    for size in [128, 256, 512]:
        x = torch.randn(1, 1, size, size)
        with torch.no_grad():
            out = model(x)
        print(f"输入 {size}x{size} → pred_masks: {out['pred_masks'].shape}")
    
    print(f"✓ 支持不同输入尺寸\n")
    
    # ==================== 测试5: 提取backbone特征 ====================
    print("[测试 5] 提取backbone特征（用于域适应）")
    print("-" * 40)
    
    # 这对域适应很有用
    images = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        backbone_features = model.backbone(images)
    
    print(f"提取到 {len(backbone_features)} 个尺度的特征:")
    for i, feat in enumerate(backbone_features):
        print(f"  Stage {i}: {feat.shape}")
    
    print(f"✓ 可以方便地提取backbone特征用于域适应\n")
    
    # ==================== 测试6: 多类别分割 ====================
    print("[测试 6] 多类别分割")
    print("-" * 40)
    
    # 创建5类分割模型
    model_multiclass = segmamba_mask2former_small(num_classes=5, num_queries=30)
    x = torch.randn(2, 1, 256, 256)
    
    with torch.no_grad():
        out = model_multiclass(x)
    
    print(f"输入: {x.shape}")
    print(f"pred_logits: {out['pred_logits'].shape}  # 6类 = 5类 + 1背景")
    print(f"pred_masks: {out['pred_masks'].shape}")
    print(f"✓ 支持多类别分割\n")
    
    print("="*80)
    print("所有测试通过！✓")
    print("="*80)
    print("\n💡 下一步:")
    print("  1. 准备您的数据集")
    print("  2. 实现DataLoader")
    print("  3. 实现训练循环（需要Hungarian matching）")
    print("  4. 开始训练！")
    print()