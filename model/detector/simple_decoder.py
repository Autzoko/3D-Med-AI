
"""
Simple Segmentation Decoder for SegMamba 2D
模仿SegMamba原版的简单上采样结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Conv + BN + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UpsampleBlock(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = ConvBNReLU(in_channels, out_channels, 3, 1, 1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


class SimpleSegDecoder(nn.Module):
    """
    简单的分割Decoder，模仿SegMamba原版结构
    
    输入：Backbone的4个stage输出 [C1, C2, C3, C4]
    输出：(B, num_classes, H, W)
    """
    def __init__(
        self,
        encoder_dims=[48, 96, 192, 384],  # SegMamba small的输出维度
        decoder_dim=256,
        num_classes=1,
        dropout=0.1,
    ):
        super().__init__()
        
        # Lateral connections（横向连接）
        self.lateral4 = nn.Conv2d(encoder_dims[3], decoder_dim, 1)
        self.lateral3 = nn.Conv2d(encoder_dims[2], decoder_dim, 1)
        self.lateral2 = nn.Conv2d(encoder_dims[1], decoder_dim, 1)
        self.lateral1 = nn.Conv2d(encoder_dims[0], decoder_dim, 1)
        
        # Smooth layers（平滑层）
        self.smooth4 = ConvBNReLU(decoder_dim, decoder_dim)
        self.smooth3 = ConvBNReLU(decoder_dim, decoder_dim)
        self.smooth2 = ConvBNReLU(decoder_dim, decoder_dim)
        self.smooth1 = ConvBNReLU(decoder_dim, decoder_dim)
        
        # Final segmentation head
        self.seg_head = nn.Sequential(
            ConvBNReLU(decoder_dim, decoder_dim, 3, 1, 1),
            nn.Dropout2d(dropout),
            ConvBNReLU(decoder_dim, decoder_dim // 2, 3, 1, 1),
            nn.Conv2d(decoder_dim // 2, num_classes, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        Args:
            features: list of [f1, f2, f3, f4]
                f1: (B, 48, H/4, W/4)
                f2: (B, 96, H/8, W/8)
                f3: (B, 192, H/16, W/16)
                f4: (B, 384, H/32, W/32)
        
        Returns:
            seg_out: (B, num_classes, H, W)
        """
        f1, f2, f3, f4 = features
        
        # Top-down pathway（自顶向下融合）
        # Stage 4 (最小的feature map)
        p4 = self.lateral4(f4)  # (B, 256, H/32, W/32)
        p4 = self.smooth4(p4)
        
        # Stage 3
        p3 = self.lateral3(f3)  # (B, 256, H/16, W/16)
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode='bilinear', align_corners=False)
        p3 = self.smooth3(p3)
        
        # Stage 2
        p2 = self.lateral2(f2)  # (B, 256, H/8, W/8)
        p2 = p2 + F.interpolate(p3, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.smooth2(p2)
        
        # Stage 1
        p1 = self.lateral1(f1)  # (B, 256, H/4, W/4)
        p1 = p1 + F.interpolate(p2, size=p1.shape[-2:], mode='bilinear', align_corners=False)
        p1 = self.smooth1(p1)
        
        # Final upsampling and segmentation
        out = self.seg_head(p1)  # (B, num_classes, H/4, W/4)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)  # (B, num_classes, H, W)
        
        return out


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    # 创建decoder
    decoder = SimpleSegDecoder(
        encoder_dims=[48, 96, 192, 384],
        decoder_dim=256,
        num_classes=1
    )
    
    # 模拟backbone输出
    B = 2
    H, W = 256, 256
    f1 = torch.randn(B, 48, H//4, W//4)     # 64x64
    f2 = torch.randn(B, 96, H//8, W//8)     # 32x32
    f3 = torch.randn(B, 192, H//16, W//16)  # 16x16
    f4 = torch.randn(B, 384, H//32, W//32)  # 8x8
    
    features = [f1, f2, f3, f4]
    
    # 前向传播
    out = decoder(features)
    
    print(f"✓ Decoder output shape: {out.shape}")  # (2, 1, 256, 256)
    print(f"✓ Parameters: {sum(p.numel() for p in decoder.parameters()) / 1e6:.2f}M")
    
    # 计算loss（示例）
    gt_mask = torch.randint(0, 2, (B, 1, H, W)).float()
    
    # BCE Loss
    loss_bce = F.binary_cross_entropy_with_logits(out, gt_mask)
    
    # Dice Loss
    pred_sigmoid = out.sigmoid()
    intersection = (pred_sigmoid * gt_mask).sum()
    dice = (2 * intersection + 1) / (pred_sigmoid.sum() + gt_mask.sum() + 1)
    loss_dice = 1 - dice
    
    total_loss = loss_bce + 2.0 * loss_dice
    
    print(f"✓ BCE Loss: {loss_bce.item():.4f}")
    print(f"✓ Dice Loss: {loss_dice.item():.4f}")
    print(f"✓ Total Loss: {total_loss.item():.4f}")
    
    print("\n✅ All tests passed!")