"""Real-ESRGAN (RRDBNet) x4 super-resolution for the optional "Enhance" mode.

The architecture is the standard RRDBNet from "ESRGAN" (Wang et al., 2018) and
"Real-ESRGAN" (Wang et al., 2021), reimplemented here so the project needs no
heavy ``basicsr`` dependency. It loads the official RealESRGAN_x4 weights
(``weights/realesrgan_x4.pth``), which are a plain RRDBNet state dict.

Unlike the project's own U-Net, this network *adds* detail: it upscales 4x and
reconstructs edges and texture, so output looks sharper, not just cleaner. It is
heavy on a CPU, so the engine processes the image in small overlapping tiles and
caps the input size. This is a local / demo feature; it is too slow for the free
hosting tiers.

Weights are read with ``weights_only=True`` so no code is executed on load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from config import PROJECT_ROOT
from utils.image_utils import to_pil, to_tensor

WEIGHTS = PROJECT_ROOT / "weights" / "realesrgan_x4.pth"
SCALE = 4


class ResidualDenseBlock(nn.Module):
    """Five-conv dense block; output is scaled by 0.2 and added to the input."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-residual dense block: three RDBs, scaled and residual-added."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet generator with 4x nearest-neighbour upsampling (Real-ESRGAN layout)."""

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, num_feat: int = 64,
                 num_block: int = 23, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def weights_path() -> Path:
    return WEIGHTS


def load_model() -> RRDBNet:
    """Load the RRDBNet with the official x4 weights, ready for inference on CPU."""
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "params_ema" in state:
        state = state["params_ema"]
    elif isinstance(state, dict) and "params" in state:
        state = state["params"]
    model = RRDBNet()
    model.load_state_dict(state, strict=True)
    return model.eval()


@torch.inference_mode()
def _sr_tiled(model: RRDBNet, x: torch.Tensor, tile: int = 160, overlap: int = 16) -> torch.Tensor:
    """Run the x4 network over ``x`` in overlapping tiles and stitch the result.

    Tiling keeps peak memory flat regardless of image size: only one tile's
    activations exist at a time. Overlap is cropped away so seams are invisible.
    """
    _, _, h, w = x.shape
    out = x.new_zeros((1, 3, h * SCALE, w * SCALE))
    step = tile - overlap
    for top in range(0, h, step):
        for left in range(0, w, step):
            t0, l0 = max(top - overlap, 0), max(left - overlap, 0)
            t1, l1 = min(top + tile + overlap, h), min(left + tile + overlap, w)
            patch = model(x[:, :, t0:t1, l0:l1])
            # Where this patch lands in the output, and which part of it to keep.
            dt, dl = (top - t0) * SCALE, (left - l0) * SCALE
            bottom, right = min(top + tile, h), min(left + tile, w)
            ph, pw = (bottom - top) * SCALE, (right - left) * SCALE
            out[:, :, top * SCALE:bottom * SCALE, left * SCALE:right * SCALE] = patch[:, :, dt:dt + ph, dl:dl + pw]
            if right >= w:
                break
        if bottom >= h:
            break
    return out.clamp_(0, 1)


class ESRGANModel:
    """Thin wrapper holding a loaded RRDBNet and running it with CPU-friendly tiling."""

    def __init__(self, max_in: int = 768) -> None:
        self.model = load_model()
        self.max_in = max_in

    @torch.inference_mode()
    def enhance(self, image: Image.Image) -> Image.Image:
        """Return a 4x super-resolved, sharper version of an RGB image.

        Large inputs are downscaled to ``max_in`` on the long side first, so the
        4x result and the CPU time both stay bounded.
        """
        img = image.convert("RGB")
        longest = max(img.size)
        if longest > self.max_in:
            s = self.max_in / longest
            img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
        x = to_tensor(img)
        y = _sr_tiled(self.model, x)
        return to_pil(y)
