"""Residual U-Net for image restoration.

- Encoder-decoder with skip connections and bilinear upsampling (avoids the
  checkerboard artifacts that transposed convolutions produce).
- Any input size works: the input is padded to a multiple of 16 internally.
- The output is a residual on the input: ``tanh(net(x)) + x``, so the network
  only has to learn the degradation, not the whole picture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image_utils import pad_to_multiple, unpad


class DoubleConv(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Max-pool then DoubleConv (one encoder step)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Bilinear x2 upsample, concat the skip connection, then DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Residual U-Net: ``output = tanh(f(x)) + x``."""

    pad_multiple = 16

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base: int = 64) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.bottleneck = Down(base * 4, base * 8)

        self.up1 = Up(base * 8, base * 4, base * 4)
        self.up2 = Up(base * 4, base * 2, base * 2)
        self.up3 = Up(base * 2, base, base)
        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded, pad_h, pad_w = pad_to_multiple(x, self.pad_multiple)

        s1 = self.inc(padded)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        b = self.bottleneck(s3)

        u = self.up1(b, s3)
        u = self.up2(u, s2)
        u = self.up3(u, s1)
        return torch.tanh(unpad(self.outc(u), pad_h, pad_w)) + x
