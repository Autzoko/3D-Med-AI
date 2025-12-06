# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SegMamba 2D Backbone (Encoder Only) for ultrasound medical image segmentation
Enhanced with full residual connections and DropPath regularization

修改记录:
- 添加 DropPath 类（正则化）
- MambaLayer 增强为 Transformer-style residual
- 添加 drop_path_rate 参数到各个函数
- 保持所有原有函数名和类名不变
"""

from __future__ import annotations
import torch.nn as nn
import torch 
from typing import List, Tuple

from mamba_ssm import Mamba
import torch.nn.functional as F


# ============================================================================
# Drop Path (Stochastic Depth) - 新增
# ============================================================================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample for residual blocks."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None, drop_path=0.0):
        """
        2D MambaLayer with enhanced residual connections
        
        新增参数:
            drop_path: DropPath rate (default: 0.0, 向后兼容)
        """
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bimamba_type="v3",
                nslices=num_slices,
        )
        # 新增: DropPath for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        """
        Enhanced residual: x = x + DropPath(Mamba(Norm(x)))
        
        Args:
            x: input tensor of shape (B, C, H, W)
        Returns:
            output tensor of shape (B, C, H, W)
        """
        B, C = x.shape[:2]
        assert C == self.dim
        
        # 保存shortcut
        shortcut = x
        
        # For 2D: flatten spatial dimensions H*W
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        
        # (B, C, H, W) -> (B, H*W, C)
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        
        # Pre-Norm + Mamba
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        # (B, H*W, C) -> (B, C, H, W)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        
        # 增强的残差连接: shortcut + DropPath(output)
        out = shortcut + self.drop_path(out)
        
        return out

    
class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim, drop_path=0.0):
        """
        Channel MLP with residual connection
        
        新增参数:
            drop_path: DropPath rate (default: 0.0)
        """
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(mlp_dim, hidden_size, 1)
        # 新增: DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Enhanced residual: x = x + DropPath(MLP(x))
        """
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        # 增强的残差连接
        return shortcut + self.drop_path(x)


class GSC(nn.Module):
    """Gated Spatial Convolution module for 2D (保持不变，已有residual)"""
    def __init__(self, in_channels) -> None:
        super().__init__()

        self.proj = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.norm = nn.InstanceNorm2d(in_channels)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(in_channels)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.norm3 = nn.InstanceNorm2d(in_channels)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.norm4 = nn.InstanceNorm2d(in_channels)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        """已有residual connection"""
        x_residual = x 

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        
        return x + x_residual


class SegMambaBackbone2D(nn.Module):
    """
    SegMamba 2D Backbone (Encoder Only)
    
    Enhanced with full residual connections:
    - DropPath (stochastic depth) for regularization
    - Transformer-style residual in MambaLayer and MLP
    - Backward compatible with original API
    """
    
    def __init__(
        self, 
        in_chans: int = 1,
        depths: List[int] = [2, 2, 2, 2],
        dims: List[int] = [48, 96, 192, 384],
        drop_path_rate: float = 0.,  # 新增参数，默认0保持向后兼容
        layer_scale_init_value: float = 1e-6,
        out_indices: List[int] = [0, 1, 2, 3],
        return_features: str = 'all'
    ):
        """
        新增参数:
            drop_path_rate: Stochastic depth rate (0.0 = 不使用, 0.1-0.2 = 推荐用于正则化)
        
        其他参数保持不变，完全向后兼容
        """
        super().__init__()
        
        self.in_chans = in_chans
        self.depths = depths
        self.dims = dims
        self.out_indices = out_indices
        self.return_features = return_features
        self.num_stages = 4

        # Stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        # Stem and downsampling layers
        self.downsample_layers = nn.ModuleList()
        
        # Stem layer
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers.append(stem)
        
        # Downsampling layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Mamba stages
        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        
        num_slices_list = [32, 16, 8, 4]
        
        cur_depth = 0
        for i in range(4):
            # GSC
            gsc = GSC(dims[i])
            
            # Stack of Mamba layers with DropPath
            stage = nn.Sequential(
                *[MambaLayer(
                    dim=dims[i], 
                    num_slices=num_slices_list[i],
                    drop_path=dpr[cur_depth + j]  # 每层递增的drop_path
                ) for j in range(depths[i])]
            )
            
            cur_depth += depths[i]

            self.stages.append(stage)
            self.gscs.append(gsc)

        # Output normalization and MLP
        self.mlps = nn.ModuleList()
        cur_depth = 0
        for i_layer in range(4):
            layer = nn.InstanceNorm2d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            
            # MLP with DropPath
            mlp_drop_path = dpr[cur_depth + depths[i_layer] - 1]
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer], drop_path=mlp_drop_path))
            cur_depth += depths[i_layer]

    def get_feature_dims(self) -> List[int]:
        """Returns the feature dimensions at each stage"""
        return self.dims
    
    def get_downsample_ratios(self) -> List[int]:
        """Returns the downsampling ratio relative to input for each stage"""
        return [2, 4, 8, 16]

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through all stages (保持原有接口)
        
        Args:
            x: Input tensor of shape (B, in_chans, H, W)
            
        Returns:
            Tuple of feature tensors from each stage
        """
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x: torch.Tensor):
        """Forward pass (保持原有接口)"""
        features = self.forward_features(x)
        
        if self.return_features == 'last':
            return features[-1]
        elif self.return_features == 'multi_scale':
            feature_info = {
                'features': features,
                'shapes': [f.shape[2:] for f in features],
                'channels': [f.shape[1] for f in features],
                'downsample_ratios': [2**(i+1) for i in self.out_indices]
            }
            return feature_info
        else:
            return features


# ============================================================================
# Convenience functions (保持原有名称，添加drop_path_rate参数)
# ============================================================================

def segmamba_backbone_tiny(in_chans=1, return_features='all', drop_path_rate=0.0):
    """
    Tiny SegMamba backbone
    
    新增参数:
        drop_path_rate: 推荐0.05-0.1 for regularization
    """
    return SegMambaBackbone2D(
        in_chans=in_chans,
        depths=[2, 2, 2, 2],
        dims=[48, 96, 192, 384],
        drop_path_rate=drop_path_rate,
        return_features=return_features
    )


def segmamba_backbone_small(in_chans=1, return_features='all', drop_path_rate=0.0):
    """
    Small SegMamba backbone
    
    新增参数:
        drop_path_rate: 推荐0.1-0.15 for regularization (BUSI数据集)
    """
    return SegMambaBackbone2D(
        in_chans=in_chans,
        depths=[2, 2, 4, 2],
        dims=[48, 96, 192, 384],
        drop_path_rate=drop_path_rate,
        return_features=return_features
    )


def segmamba_backbone_base(in_chans=1, return_features='all', drop_path_rate=0.0):
    """
    Base SegMamba backbone
    
    新增参数:
        drop_path_rate: 推荐0.15-0.2 for regularization
    """
    return SegMambaBackbone2D(
        in_chans=in_chans,
        depths=[2, 2, 8, 2],
        dims=[64, 128, 256, 512],
        drop_path_rate=drop_path_rate,
        return_features=return_features
    )


def segmamba_backbone_large(in_chans=1, return_features='all', drop_path_rate=0.0):
    """Large SegMamba backbone"""
    return SegMambaBackbone2D(
        in_chans=in_chans,
        depths=[2, 4, 12, 2],
        dims=[96, 192, 384, 768],
        drop_path_rate=drop_path_rate,
        return_features=return_features
    )


if __name__ == "__main__":
    print("="*80)
    print("SegMamba 2D Backbone with Enhanced Residual Connections")
    print("="*80)
    
    # Test backward compatibility
    print("\n[Test 1] Backward compatibility (drop_path_rate=0)")
    backbone_old = segmamba_backbone_small(in_chans=1, return_features='all')
    x = torch.randn(2, 1, 256, 256)
    features_old = backbone_old(x)
    
    print(f"Input shape: {x.shape}")
    for i, feat in enumerate(features_old):
        print(f"  Stage {i} output: {feat.shape}")
    
    # Test with DropPath
    print("\n[Test 2] With DropPath regularization (drop_path_rate=0.1)")
    backbone_new = segmamba_backbone_small(in_chans=1, return_features='all', drop_path_rate=0.1)
    backbone_new.eval()
    with torch.no_grad():
        features_new = backbone_new(x)
    
    for i, feat in enumerate(features_new):
        print(f"  Stage {i} output: {feat.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in backbone_new.parameters())
    print(f"\nTotal parameters: {params:,}")
    
    print("\n" + "="*80)
    print("✓ Enhanced residual connections added!")
    print("✓ Backward compatible with original API!")
    print("="*80)