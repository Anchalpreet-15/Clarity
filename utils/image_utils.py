"""Image <-> tensor conversions and padding helpers.

All tensors are float32 in [0, 1] with shape (1, 3, H, W).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a float32 tensor in [0, 1], shape (1, 3, H, W)."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    arr = np.array(pil_image)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)


def to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a float tensor in [0, 1], (1, 3, H, W) or (3, H, W), to a PIL image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(arr)


def pad_to_multiple(
    tensor: torch.Tensor, multiple: int = 16
) -> Tuple[torch.Tensor, int, int]:
    """Pad the bottom/right so H and W become multiples of ``multiple``.

    Returns ``(padded_tensor, pad_h, pad_w)``; hand the same pads to ``unpad``.
    Reflect padding avoids border halos, with a replicate fallback for images
    smaller than the padding itself.
    """
    if tensor.dim() != 4:
        raise ValueError(f"expected (N,C,H,W) tensor, got shape {tuple(tensor.shape)}")
    h, w = tensor.shape[-2:]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, 0, 0
    mode = "reflect" if pad_h < h and pad_w < w else "replicate"
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode), pad_h, pad_w


def unpad(tensor: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """Reverse ``pad_to_multiple`` by cropping the added bottom/right strip."""
    if pad_h == 0 and pad_w == 0:
        return tensor
    h, w = tensor.shape[-2:]
    return tensor[..., : h - pad_h, : w - pad_w]
