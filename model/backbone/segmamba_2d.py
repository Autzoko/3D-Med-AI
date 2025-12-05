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
SegMamba 2D version for ultrasound medical image segmentation
Adapted from 3D version to process 2D images
"""

from __future__ import annotations
import torch.nn as nn
import torch 
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from ssm import Mamba
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
        self.fc1 = nn.Conv2d(hidden_size, mlp_dim, 1)  # Changed from Conv3d to Conv2d
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(mlp_dim, hidden_size, 1)  # Changed from Conv3d to Conv2d

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class GSC(nn.Module):
    """Gated Spatial Convolution module for 2D"""
    def __init__(self, in_channels) -> None:
        super().__init__()

        # All Conv3d changed to Conv2d, kernel sizes adjusted from (3,3,3) to (3,3)
        self.proj = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.norm = nn.InstanceNorm2d(in_channels)  # Changed from InstanceNorm3d
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


class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        """
        2D Mamba Encoder
        Args:
            in_chans: input channels (1 for grayscale ultrasound)
            depths: number of blocks at each stage
            dims: feature dimensions at each stage
        """
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        
        # Stem layer: Conv2d with kernel 7x7 (changed from 7x7x7)
        stem = nn.Sequential(
              nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers.append(stem)
        
        # Downsampling layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm2d(dims[i]),  # Changed from InstanceNorm3d
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),  # Changed from Conv3d
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        
        # Adjusted num_slices for 2D: these represent the number of horizontal slices
        # For typical ultrasound images, adjust based on input resolution
        # Assuming input is ~512x512 or 256x256, after downsampling:
        # Stage 0: 128 or 256 height -> use 32 or 64 slices
        # Stage 1: 64 or 128 -> use 16 or 32 slices  
        # Stage 2: 32 or 64 -> use 8 or 16 slices
        # Stage 3: 16 or 32 -> use 4 or 8 slices
        num_slices_list = [32, 16, 8, 4]  # Adjusted for 2D
        
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm2d(dims[i_layer])  # Changed from InstanceNorm3d
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        """
        Args:
            x: input tensor of shape (B, C, H, W)
        Returns:
            tuple of output features at different scales
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

    def forward(self, x):
        x = self.forward_features(x)
        return x


class SegMamba2D(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=2,  # Typically binary segmentation for ultrasound
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name="instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=2,  # Changed from 3 to 2
    ) -> None:
        """
        SegMamba for 2D medical image segmentation (e.g., ultrasound)
        
        Args:
            in_chans: input channels (typically 1 for grayscale ultrasound)
            out_chans: output channels (number of segmentation classes)
            depths: number of blocks at each encoder stage
            feat_size: feature dimensions at each stage
            spatial_dims: spatial dimensions (2 for 2D images)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.spatial_dims = spatial_dims
        
        # Mamba-based encoder
        self.vit = MambaEncoder(
            in_chans, 
            depths=depths,
            dims=feat_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )
        
        # Encoder blocks for skip connections
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # Decoder blocks
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        # Output layer
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, 
            in_channels=self.feat_size[0], 
            out_channels=self.out_chans
        )

    def forward(self, x_in):
        """
        Forward pass
        Args:
            x_in: input tensor of shape (B, C, H, W)
        Returns:
            output segmentation of shape (B, out_chans, H, W)
        """
        # Get multi-scale features from encoder
        outs = self.vit(x_in)
        
        # Process skip connections
        enc1 = self.encoder1(x_in)
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])
        
        # Decoder with skip connections
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)
                
        return self.out(out)


# Convenience function to create model
def segmamba_2d_tiny(in_chans=1, out_chans=2):
    """Create a tiny SegMamba2D model"""
    return SegMamba2D(
        in_chans=in_chans,
        out_chans=out_chans,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
    )


def segmamba_2d_small(in_chans=1, out_chans=2):
    """Create a small SegMamba2D model"""
    return SegMamba2D(
        in_chans=in_chans,
        out_chans=out_chans,
        depths=[2, 2, 4, 2],
        feat_size=[48, 96, 192, 384],
    )


def segmamba_2d_base(in_chans=1, out_chans=2):
    """Create a base SegMamba2D model"""
    return SegMamba2D(
        in_chans=in_chans,
        out_chans=out_chans,
        depths=[2, 2, 8, 2],
        feat_size=[64, 128, 256, 512],
    )


if __name__ == "__main__":
    # Test the model
    model = segmamba_2d_tiny(in_chans=1, out_chans=2)
    x = torch.randn(2, 1, 256, 256)  # Batch of 2, 1 channel, 256x256 images
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")