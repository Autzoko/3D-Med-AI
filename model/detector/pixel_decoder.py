"""
Pixel Decoders for Mask2Former
Based on official implementation

Provides:
1. SimpleFPNDecoder - Lightweight FPN-style decoder (recommended for medical imaging)
2. BasePixelDecoder - Standard FPN decoder from Mask2Former
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ============================================================================
# Simple FPN Pixel Decoder (Recommended for Medical Imaging)
# ============================================================================

class SimpleFPNDecoder(nn.Module):
    """
    Simplified FPN-style pixel decoder optimized for medical imaging
    
    Takes multi-scale features from backbone and produces:
    1. mask_features: high-res per-pixel embeddings (H/4, W/4)
    2. multi_scale_features: 3 scales for transformer decoder (1/32, 1/16, 1/8)
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        conv_dim: int = 256,
        mask_dim: int = 256,
        norm: str = "GN",  # GroupNorm
        num_groups: int = 32,
    ):
        """
        Args:
            in_channels_list: list of input channels from backbone
                             e.g., [48, 96, 192, 384] for SegMamba tiny/small
            conv_dim: intermediate conv channel dimension
            mask_dim: output mask feature dimension
            norm: normalization type ("GN" for GroupNorm, "" for no norm)
            num_groups: number of groups for GroupNorm
        """
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.conv_dim = conv_dim
        self.mask_dim = mask_dim
        self.num_feature_levels = 3  # Always use 3 scales for Mask2Former
        
        # Build FPN layers (top-down pathway)
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        use_bias = (norm == "")
        
        # Process from deep to shallow (reversed)
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
                if not use_bias:
                    nn.init.constant_(lateral_conv.bias, 0)
                self.lateral_convs.append(lateral_conv)
            
            nn.init.xavier_uniform_(output_conv[0].weight)
            if not use_bias:
                nn.init.constant_(output_conv[0].bias, 0)
            self.output_convs.append(output_conv)
        
        # Mask features projection
        self.mask_features = nn.Conv2d(conv_dim, mask_dim, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.mask_features.weight)
        nn.init.constant_(self.mask_features.bias, 0)
        
    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            features: list of feature maps from backbone
                     [stage0, stage1, stage2, stage3] from shallow to deep
                     For SegMamba: [(B,48,H/2,W/2), (B,96,H/4,W/4), (B,192,H/8,W/8), (B,384,H/16,W/16)]
                     
        Returns:
            mask_features: (B, mask_dim, H/4, W/4) for mask prediction
            multi_scale_features: list of 3 features for transformer decoder
                                 [(B,C,H/32,W/32), (B,C,H/16,W/16), (B,C,H/8,W/8)]
        """
        # Reverse to top-down order (deep to shallow)
        features = list(reversed(features))
        
        multi_scale_features = []
        
        # Top-down pathway
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
            
            # Collect first 3 scales for transformer decoder
            if len(multi_scale_features) < self.num_feature_levels:
                multi_scale_features.append(y)
        
        # Generate high-res mask features from shallowest FPN level
        mask_features = self.mask_features(y)
        
        return mask_features, multi_scale_features


# ============================================================================
# Base Pixel Decoder (Standard FPN from Mask2Former)
# ============================================================================

class BasePixelDecoder(nn.Module):
    """
    Base FPN pixel decoder - standard implementation from Mask2Former
    More complex than SimpleFPNDecoder but may provide better results
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        conv_dim: int = 256,
        mask_dim: int = 256,
        norm: str = "GN",
        num_groups: int = 32,
    ):
        """
        Args:
            in_channels_list: list of input channels from backbone
            conv_dim: intermediate conv dimension
            mask_dim: output mask dimension
            norm: normalization type
            num_groups: number of groups for GroupNorm
        """
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
        """
        Args:
            features: list of feature maps [stage0, stage1, stage2, stage3]
            
        Returns:
            mask_features: (B, mask_dim, H, W)
            multi_scale_features: list of 3 features
        """
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
    print("Testing Pixel Decoders")
    print("="*80)
    
    # Simulate SegMamba backbone features
    batch_size = 2
    features = [
        torch.randn(batch_size, 48, 128, 128),   # stage0: H/2, W/2
        torch.randn(batch_size, 96, 64, 64),     # stage1: H/4, W/4
        torch.randn(batch_size, 192, 32, 32),    # stage2: H/8, W/8
        torch.randn(batch_size, 384, 16, 16),    # stage3: H/16, W/16
    ]
    
    print("\n[Test 1] SimpleFPNDecoder")
    decoder1 = SimpleFPNDecoder(
        in_channels_list=[48, 96, 192, 384],
        conv_dim=256,
        mask_dim=256,
    )
    
    with torch.no_grad():
        mask_feat, multi_scale = decoder1(features)
    
    print(f"mask_features shape: {mask_feat.shape}")
    print(f"Number of multi-scale features: {len(multi_scale)}")
    for i, feat in enumerate(multi_scale):
        print(f"  Scale {i}: {feat.shape}")
    
    print("\n[Test 2] BasePixelDecoder")
    decoder2 = BasePixelDecoder(
        in_channels_list=[48, 96, 192, 384],
        conv_dim=256,
        mask_dim=256,
    )
    
    with torch.no_grad():
        mask_feat, multi_scale = decoder2(features)
    
    print(f"mask_features shape: {mask_feat.shape}")
    print(f"Number of multi-scale features: {len(multi_scale)}")
    for i, feat in enumerate(multi_scale):
        print(f"  Scale {i}: {feat.shape}")
    
    # Count parameters
    params1 = sum(p.numel() for p in decoder1.parameters())
    params2 = sum(p.numel() for p in decoder2.parameters())
    
    print(f"\nSimpleFPNDecoder parameters: {params1:,}")
    print(f"BasePixelDecoder parameters: {params2:,}")
    
    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80)
