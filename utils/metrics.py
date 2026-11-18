"""Image quality metrics: PSNR and SSIM.

Both accept PIL images or torch tensors in [0, 1]. They coerce to NumPy
arrays with values in [0, 1] before computing.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as _ssim


ImageLike = Union[Image.Image, torch.Tensor, np.ndarray]


def _to_array(img: ImageLike) -> np.ndarray:
    """Return a float32 NumPy array in [0, 1] with shape (H, W, 3)."""
    if isinstance(img, Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr
    if isinstance(img, torch.Tensor):
        t = img.detach().cpu().float()
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)
        return t.clamp(0.0, 1.0).numpy()
    if isinstance(img, np.ndarray):
        arr = img.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        return arr
    raise TypeError(f"unsupported image type: {type(img).__name__}")


def psnr(img1: ImageLike, img2: ImageLike, max_val: float = 1.0) -> float:
    """Peak signal-to-noise ratio in decibels (higher is better)."""
    a = _to_array(img1)
    b = _to_array(img2)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch for PSNR: {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))


def ssim(img1: ImageLike, img2: ImageLike) -> float:
    """Structural similarity index in [0, 1] (higher is better)."""
    a = _to_array(img1)
    b = _to_array(img2)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch for SSIM: {a.shape} vs {b.shape}")
    return float(_ssim(a, b, channel_axis=-1, data_range=1.0))
