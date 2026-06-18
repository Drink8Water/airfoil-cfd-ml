"""ResU-Net: U-Net with residual blocks in each encoder/decoder stage.

Architecture:
  5-level symmetric encoder-decoder with skip connections.
  Each level uses a ResidualBlock (Conv→BN→ReLU→Conv→BN + skip → ReLU).
  Encoder downsamples with stride-2 conv; decoder upsamples with bilinear + conv.

Configurable: channel_exponent, dropout.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_model


class ResidualBlock(nn.Module):
    """Two 3×3 convs with BatchNorm + ReLU and a residual connection.

    If in_channels != out_channels, the skip connection uses a 1×1 conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class DownBlock(nn.Module):
    """ResidualBlock followed by stride-2 conv for downsampling."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, dropout)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.res(x)
        down = self.down(skip)
        return down, skip


class UpBlock(nn.Module):
    """Upsample (bilinear) + conv, then concat skip, then ResidualBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)
        self.res = ResidualBlock(out_ch + skip_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x)


@register_model("res_unet")
class ResUNet(nn.Module):
    """ResU-Net for airfoil flow surrogate modelling.

    Args:
        in_channels: input channel count (depends on geometry_mode).
        channel_exponent: base channels = 2**ce (default 6 → 64).
        dropout: dropout rate in residual blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channel_exponent: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        ch = int(2 ** channel_exponent + 0.5)

        # Encoder
        self.enc1 = DownBlock(in_channels, ch, dropout)
        self.enc2 = DownBlock(ch, ch * 2, dropout)
        self.enc3 = DownBlock(ch * 2, ch * 2, dropout)
        self.enc4 = DownBlock(ch * 2, ch * 4, dropout)
        self.enc5 = DownBlock(ch * 4, ch * 8, dropout)

        # Bottleneck
        self.bottleneck = ResidualBlock(ch * 8, ch * 8, dropout)

        # Decoder
        self.dec5 = UpBlock(ch * 8, ch * 8, ch * 4, dropout)
        self.dec4 = UpBlock(ch * 4, ch * 4, ch * 2, dropout)
        self.dec3 = UpBlock(ch * 2, ch * 2, ch * 2, dropout)
        self.dec2 = UpBlock(ch * 2, ch * 2, ch, dropout)
        self.dec1 = UpBlock(ch, ch, ch, dropout)

        # Output
        self.head = nn.Conv2d(ch, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        d1, s1 = self.enc1(x)
        d2, s2 = self.enc2(d1)
        d3, s3 = self.enc3(d2)
        d4, s4 = self.enc4(d3)
        d5, s5 = self.enc5(d4)

        # Bottleneck
        b = self.bottleneck(d5)

        # Decoder
        u5 = self.dec5(b, s5)
        u4 = self.dec4(u5, s4)
        u3 = self.dec3(u4, s3)
        u2 = self.dec2(u3, s2)
        u1 = self.dec1(u2, s1)

        return self.head(u1)
