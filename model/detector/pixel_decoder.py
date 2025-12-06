"""
Pixel Decoders for Mask2Former
Based on official implementation

Enhanced with encoder-decoder skip connections

修改记录:
- SimpleFPNDecoder 添加 use_skip_connections 参数（默认False保持向后兼容）
- 添加 skip_convs 和 fusion_weights 用于encoder-decoder残差连接
- 保持原有类名和函数接口完全不变
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ============================================================================
# Simple FPN Pixel Decoder with Skip Connections
# ============================================================================

class SimpleFPNDecoder(nn.Module):
    """
    Simplified FPN-style pixel decoder with optional encoder-decoder skip connections
    
    Enhanced features:
    - Original FPN top-down pathway
    - Optional skip connections from encoder features (类似U-Net)
    - Learnable fusion weights between FPN and encoder features
    
    完全向后兼容：use_skip_connections=False 时行为与原版完全一致
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        conv_dim: int = 256,
        mask_dim: int = 256,
        norm: str = "GN",
        num_groups: int = 32,
        use_skip_connections: bool = False,  # 新增参数，默认False保持向后兼容
    ):
        """
        Args:
            in_channels_list: list of input channels from backbone
                             e.g., [48, 96, 192, 384] for SegMamba tiny/small
            conv_dim: intermediate conv channel dimension
            mask_dim: output mask feature dimension
            norm: normalization type ("GN" for GroupNorm, "" for no norm)
            num_groups: number of groups for GroupNorm
            use_skip_connections: 是否使用encoder-decoder skip connections (新增)
        """
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.conv_dim = conv_dim
        self.mask_dim = mask_dim
        self.num_feature_levels = 3
        self.use_skip_connections = use_skip_connections  # 新增
        
        use_bias = (norm == "")
        
        # ===== 原有的FPN layers =====
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        for idx, in_channels in enumerate(reversed(in_channels_list)):
            if idx == 0:
                # Deepest level - no lateral conv
                output_conv = nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=3, padding=1, bias=use_bias),
                    nn.GroupNorm(num_groups, conv_dim) if norm == "GN" else nn.Identity(),
                    nn.ReLU(inplace=True)
                )
                self.lateral_convs.append(None)
            else:
                # Other levels - lateral + output conv
                lateral_conv = nn.Conv2d(in_channels, conv_dim, kernel_size=1, bias=use_bias)
                output_conv = nn.Sequential(
                    nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1, bias=use_bias),
                    nn.GroupNorm(num_groups, conv_dim) if norm == "GN" else nn.Identity(),
                    nn.ReLU(inplace=True)
                )
                nn.init.xavier_uniform_(lateral_conv.weight)
                if use_bias:
                    nn.init.constant_(lateral_conv.bias, 0)
                self.lateral_convs.append(lateral_conv)
            
            nn.init.xavier_uniform_(output_conv[0].weight)
            if use_bias:
                nn.init.constant_(output_conv[0].bias, 0)
            self.output_convs.append(output_conv)
        
        # ===== 新增: Skip connection modules =====
        if use_skip_connections:
            self.skip_convs = nn.ModuleList()
            
            # 为每个scale创建skip connection projection
            for in_channels in reversed(in_channels_list):
                skip_conv = nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups, conv_dim) if norm == "GN" else nn.Identity(),
                )
                nn.init.xavier_uniform_(skip_conv[0].weight)
                self.skip_convs.append(skip_conv)
            
            # Learnable fusion weights (FPN vs Skip)
            # 初始化为[0.7, 0.3]，更信任FPN路径
            self.fusion_weights = nn.ParameterList([
                nn.Parameter(torch.tensor([0.7, 0.3]))
                for _ in range(len(in_channels_list))
            ])
        
        # ===== Mask features projection =====
        self.mask_features = nn.Conv2d(conv_dim, mask_dim, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.mask_features.weight)
        nn.init.constant_(self.mask_features.bias, 0)
        
    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            features: list of feature maps from backbone
                     [stage0, stage1, stage2, stage3] from shallow to deep
                     
        Returns:
            mask_features: (B, mask_dim, H/4, W/4)
            multi_scale_features: list of 3 features for transformer decoder
        """
        # 保存原始encoder features（用于skip connections）
        encoder_features = features if self.use_skip_connections else None
        
        # Reverse to top-down order (deep to shallow)
        features = list(reversed(features))
        
        multi_scale_features = []
        
        # ===== Top-down pathway with optional skip connections =====
        for idx, (feat, lateral_conv, output_conv) in enumerate(
            zip(features, self.lateral_convs, self.output_convs)
        ):
            if lateral_conv is None:
                # Deepest level
                y = output_conv(feat)
            else:
                # Upsample previous level and add lateral connection
                lateral = lateral_conv(feat)
                y = lateral + F.interpolate(y, size=lateral.shape[-2:], mode="nearest")
                y = output_conv(y)
            
            # ===== 新增: Encoder-Decoder Skip Connection =====
            if self.use_skip_connections:
                # Process encoder feature
                skip_feat = self.skip_convs[idx](features[idx])
                
                # Learnable weighted fusion: w1*FPN + w2*Skip
                w = F.softmax(self.fusion_weights[idx], dim=0)
                y = w[0] * y + w[1] * skip_feat  # ✅ Encoder-decoder residual!
            
            # Collect multi-scale features
            if len(multi_scale_features) < self.num_feature_levels:
                multi_scale_features.append(y)
        
        # Mask features
        mask_features = self.mask_features(y)
        
        return mask_features, multi_scale_features


# ============================================================================
# Base Pixel Decoder (保持不变)
# ============================================================================

class BasePixelDecoder(nn.Module):
    """Base FPN pixel decoder - standard implementation from Mask2Former"""
    
    def __init__(
        self,
        in_channels_list: List[int],
        conv_dim: int = 256,
        mask_dim: int = 256,
        norm: str = "GN",
        num_groups: int = 32,
    ):
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.conv_dim = conv_dim
        self.mask_dim = mask_dim
        self.maskformer_num_feature_levels = 3
        
        # Build lateral and output convs
        lateral_convs = []
        output_convs = []
        
        use_bias = (norm == "")
        
        for idx, in_channels in enumerate(in_channels_list):
            if idx == len(in_channels_list) - 1:
                # Deepest level
                output_norm = nn.GroupNorm(num_groups, conv_dim) if norm == "GN" else nn.Identity()
                output_conv = nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    output_norm,
                    nn.ReLU(inplace=True)
                )
                nn.init.xavier_uniform_(output_conv[0].weight)
                if not use_bias:
                    nn.init.constant_(output_conv[0].bias, 0)
                
                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                # Other levels
                lateral_norm = nn.GroupNorm(num_groups, conv_dim) if norm == "GN" else nn.Identity()
                output_norm = nn.GroupNorm(num_groups, conv_dim) if norm == "GN" else nn.Identity()
                
                lateral_conv = nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1, bias=use_bias),
                    lateral_norm
                )
                output_conv = nn.Sequential(
                    nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    output_norm,
                    nn.ReLU(inplace=True)
                )
                
                nn.init.xavier_uniform_(lateral_conv[0].weight)
                nn.init.xavier_uniform_(output_conv[0].weight)
                if not use_bias:
                    nn.init.constant_(lateral_conv[0].bias, 0)
                    nn.init.constant_(output_conv[0].bias, 0)
                
                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        
        # Reverse for top-down computation
        self.lateral_convs = nn.ModuleList(lateral_convs[::-1])
        self.output_convs = nn.ModuleList(output_convs[::-1])
        
        # Mask features
        self.mask_features = nn.Conv2d(conv_dim, mask_dim, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.mask_features.weight)
        nn.init.constant_(self.mask_features.bias, 0)
    
    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass"""
        multi_scale_features = []
        num_cur_levels = 0
        
        # Reverse to top-down order
        features = list(reversed(features))
        
        for idx, (feat, lateral_conv, output_conv) in enumerate(
            zip(features, self.lateral_convs, self.output_convs)
        ):
            if lateral_conv is None:
                y = output_conv(feat)
            else:
                cur_fpn = lateral_conv(feat)
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
            
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        
        return self.mask_features(y), multi_scale_features


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Testing Enhanced Pixel Decoders with Skip Connections")
    print("="*80)
    
    # Simulate SegMamba backbone features
    batch_size = 2
    features = [
        torch.randn(batch_size, 48, 128, 128),   # stage0
        torch.randn(batch_size, 96, 64, 64),     # stage1
        torch.randn(batch_size, 192, 32, 32),    # stage2
        torch.randn(batch_size, 384, 16, 16),    # stage3
    ]
    
    print("\n[Test 1] SimpleFPNDecoder - Original (no skip)")
    decoder_original = SimpleFPNDecoder(
        in_channels_list=[48, 96, 192, 384],
        conv_dim=256,
        mask_dim=256,
        use_skip_connections=False,  # 原版模式
    )
    
    with torch.no_grad():
        mask_feat, multi_scale = decoder_original(features)
    
    print(f"mask_features: {mask_feat.shape}")
    print(f"multi_scale features: {[f.shape for f in multi_scale]}")
    
    print("\n[Test 2] SimpleFPNDecoder - With Skip Connections")
    decoder_skip = SimpleFPNDecoder(
        in_channels_list=[48, 96, 192, 384],
        conv_dim=256,
        mask_dim=256,
        use_skip_connections=True,  # ✅ 启用skip connections
    )
    
    with torch.no_grad():
        mask_feat, multi_scale = decoder_skip(features)
    
    print(f"mask_features: {mask_feat.shape}")
    print(f"multi_scale features: {[f.shape for f in multi_scale]}")
    
    # Check fusion weights
    print("\nLearnable fusion weights:")
    for i, w in enumerate(decoder_skip.fusion_weights):
        w_softmax = F.softmax(w, dim=0)
        print(f"  Level {i}: FPN={w_softmax[0].item():.3f}, Skip={w_softmax[1].item():.3f}")
    
    # Count parameters
    params_original = sum(p.numel() for p in decoder_original.parameters())
    params_skip = sum(p.numel() for p in decoder_skip.parameters())
    
    print(f"\nParameters:")
    print(f"  Original: {params_original:,}")
    print(f"  With Skip: {params_skip:,}")
    print(f"  Increase: {params_skip - params_original:,} (+{(params_skip/params_original - 1)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("✓ Encoder-decoder skip connections added!")
    print("✓ Backward compatible (use_skip_connections=False)")
    print("="*80)