"""PyTorch datasets for training and validating the restoration model.

- ``RestorationDataset``: fixed (degraded, clean) pairs on disk, ``clean/`` + ``degraded/``
  folders. Used for validation and the held-out test set.
- ``OnTheFlyDataset``: random crops of clean photos, degraded freshly every time one is drawn.
- ``RealPatchDataset``: random crops from real (noisy, clean) camera pairs stored in a
  ``.npy`` file (see ``scripts/prepare_real_data.py``).
- ``MixedDataset``: draws from the synthetic and real datasets with a fixed probability.

Tensors are float32 in [0, 1], shape (3, H, W).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.degradations import random_degrade

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _load(path: Path) -> np.ndarray:
    """Read an image as an RGB uint8 array."""
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float().div_(255.0)


def _crop_and_flip(arrays: Sequence[np.ndarray], size: int, augment: bool) -> List[np.ndarray]:
    """Take the same random ``size`` crop from every array, with optional shared flips."""
    height, width = arrays[0].shape[:2]
    if height < size or width < size:
        pad = ((0, max(0, size - height)), (0, max(0, size - width)), (0, 0))
        arrays = [np.pad(a, pad, mode="reflect") for a in arrays]
        height, width = arrays[0].shape[:2]
    top, left = random.randint(0, height - size), random.randint(0, width - size)
    out = [a[top : top + size, left : left + size] for a in arrays]
    if augment:
        if random.random() < 0.5:
            out = [a[:, ::-1] for a in out]
        if random.random() < 0.5:
            out = [a[::-1] for a in out]
    return out


def list_images(folder: Path) -> List[Path]:
    """Image files directly inside ``folder``, sorted by name."""
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)


class RestorationDataset(Dataset):
    """Fixed paired clean/degraded images (whole images, or random crops)."""

    def __init__(
        self,
        root_dir: str | Path,
        patch_size: int = 128,
        augment: bool = True,
        full_image: bool = False,
    ) -> None:
        """
        Args:
            root_dir: folder containing ``clean/`` and ``degraded/``.
            patch_size: crop size when ``full_image`` is False.
            augment: apply random flips to crops.
            full_image: return whole images instead of random crops.
        """
        root = Path(root_dir)
        clean_dir, degraded_dir = root / "clean", root / "degraded"
        if not clean_dir.is_dir() or not degraded_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {clean_dir} and {degraded_dir}. Run `python data/prepare_data.py` first."
            )
        degraded_names = {p.name for p in degraded_dir.iterdir()}
        self.pairs: List[Tuple[Path, Path]] = [
            (degraded_dir / p.name, p) for p in list_images(clean_dir) if p.name in degraded_names
        ]
        if not self.pairs:
            raise RuntimeError(f"No paired images found under {root}")
        self.patch_size, self.augment, self.full_image = patch_size, augment, full_image

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        degraded_path, clean_path = self.pairs[index]
        degraded, clean = _load(degraded_path), _load(clean_path)
        height, width = min(degraded.shape[0], clean.shape[0]), min(degraded.shape[1], clean.shape[1])
        degraded, clean = degraded[:height, :width], clean[:height, :width]
        if not self.full_image:
            degraded, clean = _crop_and_flip([degraded, clean], self.patch_size, self.augment)
        return _to_tensor(degraded), _to_tensor(clean)


class OnTheFlyDataset(Dataset):
    """Crops of clean photos, degraded with a fresh random recipe on every draw."""

    def __init__(self, folders: Sequence[str | Path], patch_size: int = 128, length: int = 4000) -> None:
        """
        Args:
            folders: directories of clean photos (all are pooled).
            patch_size: crop size.
            length: number of samples that make up one pass over this dataset.
        """
        self.images = [_load(p) for folder in folders for p in list_images(Path(folder))]
        if not self.images:
            raise RuntimeError(f"No clean photos found in {list(map(str, folders))}")
        self.patch_size, self.length = patch_size, length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = random.choice(self.images)
        (clean,) = _crop_and_flip([clean], self.patch_size, augment=True)
        rng = np.random.default_rng(random.getrandbits(64))
        return _to_tensor(random_degrade(clean, rng)), _to_tensor(clean)


class RealPatchDataset(Dataset):
    """Random crops from real noisy/clean camera pairs stored as (N, 2, P, P, 3) uint8."""

    def __init__(self, npy_path: str | Path, patch_size: int = 128, length: int = 4000) -> None:
        self.patches = np.load(npy_path, mmap_mode="r")
        self.patch_size, self.length = patch_size, length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair = self.patches[random.randrange(len(self.patches))]
        noisy, clean = _crop_and_flip([np.asarray(pair[0]), np.asarray(pair[1])], self.patch_size, augment=True)
        return _to_tensor(noisy), _to_tensor(clean)


class RealPairDataset(Dataset):
    """Whole real noisy/clean image pairs: ``*_real.JPG`` with its ``*_mean.JPG`` reference."""

    def __init__(self, folder: str | Path) -> None:
        self.pairs = [
            (p, p.with_name(p.name.replace("_real", "_mean")))
            for p in sorted(Path(folder).glob("*_real.JPG"))
            if p.with_name(p.name.replace("_real", "_mean")).exists()
        ]
        if not self.pairs:
            raise RuntimeError(f"No *_real.JPG / *_mean.JPG pairs in {folder}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy, clean = self.pairs[index]
        return _to_tensor(_load(noisy)), _to_tensor(_load(clean))


class MixedDataset(Dataset):
    """Draws from ``real`` with probability ``real_fraction``, otherwise from ``synthetic``."""

    def __init__(self, synthetic: Dataset, real: Dataset, real_fraction: float, length: int) -> None:
        self.synthetic, self.real = synthetic, real
        self.real_fraction, self.length = real_fraction, length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        source = self.real if random.random() < self.real_fraction else self.synthetic
        return source[index]
