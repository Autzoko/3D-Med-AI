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
This is the encoder-only version that can be used with custom decoders
"""

from __future__ import annotations
import torch.nn as nn
import torch 
from typing import List, Tuple

from mamba_ssm import Mamba
import torch.nn.functional as F 


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
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        """
        2D MambaLayer for processing 2D feature maps
        Args:
            dim: channel dimension
            d_state: SSM state expansion factor
            d_conv: Local convolution width
            expand: Block expansion factor
            num_slices: number of slices for spatial direction processing (height slices in 2D)
        """
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim,  # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices=num_slices,
        )
    
    def forward(self, x):
        """
        Args:
            x: input tensor of shape (B, C, H, W)
        Returns:
            output tensor of shape (B, C, H, W)
        """
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        
        # For 2D: flatten spatial dimensions H*W
        n_tokens = x.shape[2:].numel()  # H * W
        img_dims = x.shape[2:]  # (H, W)
        
        # Reshape from (B, C, H, W) to (B, C, H*W) then transpose to (B, H*W, C)
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        # Reshape back to (B, C, H, W)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip
        
        return out

    
class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class GSC(nn.Module):
    """Gated Spatial Convolution module for 2D"""
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
        """
        Args:
            x: input tensor of shape (B, C, H, W)
        """
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
    
    This module extracts multi-scale features from 2D images using the SegMamba architecture.
    It can be used as a standalone backbone and connected to custom decoders.
    
    Features:
    - Multi-scale feature extraction at 4 different resolutions
    - Tri-orientated Mamba (ToM) for global context modeling
    - Gated Spatial Convolution (GSC) for local feature enhancement
    - Skip connections preservation for U-Net style architectures
    """
    
    def __init__(
        self, 
        in_chans: int = 1,
        depths: List[int] = [2, 2, 2, 2],
        dims: List[int] = [48, 96, 192, 384],
        drop_path_rate: float = 0.,
        layer_scale_init_value: float = 1e-6,
        out_indices: List[int] = [0, 1, 2, 3],
        return_features: str = 'all'  # 'all', 'last', or 'multi_scale'
    ):
        """
        Args:
            in_chans: Number of input channels (1 for grayscale ultrasound)
            depths: Number of Mamba blocks at each stage
            dims: Feature dimensions at each stage [stage0, stage1, stage2, stage3]
            drop_path_rate: Stochastic depth rate (not used in current implementation)
            layer_scale_init_value: Initial value for layer scale (not used in current implementation)
            out_indices: Which stages to output (default: all stages)
            return_features: 
                - 'all': return all intermediate features as tuple
                - 'last': return only the last (deepest) feature
                - 'multi_scale': return features with spatial sizes for decoder
        """
        super().__init__()
        
        self.in_chans = in_chans
        self.depths = depths
        self.dims = dims
        self.out_indices = out_indices
        self.return_features = return_features
        self.num_stages = 4

        # Stem and downsampling layers
        self.downsample_layers = nn.ModuleList()
        
        # Stem layer: 7x7 conv with stride 2
        # Input: (B, in_chans, H, W) -> Output: (B, dims[0], H/2, W/2)
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers.append(stem)
        
        # Downsampling layers for stages 1-3
        # Each: (B, dims[i], H, W) -> (B, dims[i+1], H/2, W/2)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Mamba stages
        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        
        # num_slices for different stages
        # Adjust based on expected input resolution
        # Default assumes ~256x256 input
        num_slices_list = [32, 16, 8, 4]
        
        for i in range(4):
            # Gated Spatial Convolution
            gsc = GSC(dims[i])
            
            # Stack of Mamba layers
            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)

        # Output normalization and MLP for each stage
        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm2d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def get_feature_dims(self) -> List[int]:
        """
        Returns the feature dimensions at each stage
        Useful for building decoders
        """
        return self.dims
    
    def get_downsample_ratios(self) -> List[int]:
        """
        Returns the downsampling ratio relative to input for each stage
        Stage 0: 2x, Stage 1: 4x, Stage 2: 8x, Stage 3: 16x
        """
        return [2, 4, 8, 16]

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through all stages
        
        Args:
            x: Input tensor of shape (B, in_chans, H, W)
            
        Returns:
            Tuple of feature tensors from each stage specified in out_indices
            Each feature has shape (B, dims[i], H/(2^(i+1)), W/(2^(i+1)))
        """
        outs = []
        for i in range(4):
            # Downsample
            x = self.downsample_layers[i](x)
            # Gated Spatial Conv
            x = self.gscs[i](x)
            # Mamba blocks
            x = self.stages[i](x)

            # Apply normalization and MLP if this stage is in out_indices
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x: torch.Tensor):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, in_chans, H, W)
            
        Returns:
            Depends on return_features setting:
            - 'all': tuple of all stage features
            - 'last': only the last (deepest) feature
            - 'multi_scale': dict with 'features' and 'shapes'
        """
        features = self.forward_features(x)
        
        if self.return_features == 'last':
            return features[-1]
        elif self.return_features == 'multi_scale':
            # Return features with their spatial sizes
            feature_info = {
                'features': features,
                'shapes': [f.shape[2:] for f in features],
                'channels': [f.shape[1] for f in features],
                'downsample_ratios': [2**(i+1) for i in self.out_indices]
            }
            return feature_info
        else:  # 'all'
            return features


# Convenience functions for different model sizes

def segmamba_backbone_tiny(in_chans=1, return_features='all'):
    """
    Tiny SegMamba backbone
    Parameters: ~5M
    """
    return SegMambaBackbone2D(
        in_chans=in_chans,
        depths=[2, 2, 2, 2],
        dims=[48, 96, 192, 384],
        return_features=return_features
    )


def segmamba_backbone_small(in_chans=1, return_features='all'):
    """
    Small SegMamba backbone
    Parameters: ~8M
    """
    return SegMambaBackbone2D(
        in_chans=in_chans,
        depths=[2, 2, 4, 2],
        dims=[48, 96, 192, 384],
        return_features=return_features
    )


def segmamba_backbone_base(in_chans=1, return_features='all'):
    """
    Base SegMamba backbone
    Parameters: ~15M
    """
    return SegMambaBackbone2D(
        in_chans=in_chans,
        depths=[2, 2, 8, 2],
        dims=[64, 128, 256, 512],
        return_features=return_features
    )


def segmamba_backbone_large(in_chans=1, return_features='all'):
    """
    Large SegMamba backbone
    Parameters: ~25M
    """
    return SegMambaBackbone2D(
        in_chans=in_chans,
        depths=[2, 4, 12, 2],
        dims=[96, 192, 384, 768],
        return_features=return_features
    )


if __name__ == "__main__":
    print("="*80)
    print("SegMamba 2D Backbone Testing")
    print("="*80)
    
    # Test 1: Basic forward pass with all features
    print("\n[Test 1] Basic forward pass - return all features")
    backbone = segmamba_backbone_tiny(in_chans=1, return_features='all')
    x = torch.randn(2, 1, 256, 256)
    features = backbone(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Number of output features: {len(features)}")
    for i, feat in enumerate(features):
        print(f"  Stage {i} output shape: {feat.shape}")
    
    # Test 2: Return only last feature
    print("\n[Test 2] Return last feature only")
    backbone_last = segmamba_backbone_tiny(in_chans=1, return_features='last')
    last_feat = backbone_last(x)
    print(f"Last feature shape: {last_feat.shape}")
    
    # Test 3: Return multi-scale info
    print("\n[Test 3] Return multi-scale information")
    backbone_ms = segmamba_backbone_tiny(in_chans=1, return_features='multi_scale')
    feature_info = backbone_ms(x)
    print(f"Number of features: {len(feature_info['features'])}")
    print(f"Feature channels: {feature_info['channels']}")
    print(f"Feature shapes: {feature_info['shapes']}")
    print(f"Downsample ratios: {feature_info['downsample_ratios']}")
    
    # Test 4: Different input sizes
    print("\n[Test 4] Different input resolutions")
    for size in [128, 256, 512]:
        x_test = torch.randn(1, 1, size, size)
        feats = backbone(x_test)
        print(f"Input size {size}x{size}:")
        for i, feat in enumerate(feats):
            print(f"  Stage {i}: {feat.shape}")
    
    # Test 5: Model sizes comparison
    print("\n[Test 5] Model size comparison")
    models = {
        'Tiny': segmamba_backbone_tiny(),
        'Small': segmamba_backbone_small(),
        'Base': segmamba_backbone_base(),
        'Large': segmamba_backbone_large(),
    }
    
    for name, model in models.items():
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{name:10s}: {num_params:>12,} parameters")
    
    # Test 6: Feature dimension info
    print("\n[Test 6] Feature dimensions for each model")
    for name, model in models.items():
        dims = model.get_feature_dims()
        ratios = model.get_downsample_ratios()
        print(f"{name:10s}: dims={dims}, downsample_ratios={ratios}")
    
    print("\n" + "="*80)
    print("All tests completed successfully!")
    print("="*80)