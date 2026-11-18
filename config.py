"""Central configuration: hyperparameters and paths shared by training and inference."""

from __future__ import annotations

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent


class Config:
    """Every tunable setting in one place."""

    # Training
    epochs = 100
    batch_size = 8
    lr = 1e-4
    patch_size = 128
    num_workers = 2
    val_every = 5

    # Paths (relative to the project root)
    data_dir = PROJECT_ROOT / "data"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    checkpoint_dir = PROJECT_ROOT / "checkpoints"

    # Model
    base_channels = 64

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Degradation used to create training pairs
    noise_sigma_min = 10
    noise_sigma_max = 50
    blur_kernel_min = 5
    blur_kernel_max = 15
    blur_sigma_min = 1.0
    blur_sigma_max = 4.0

    # Dataset split sizes (train / val / test)
    train_split = 360
    val_split = 40
    test_split = 68

    # NAFNet needs inputs padded to a multiple of this
    pad_multiple_nafnet = 32

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create the data and checkpoint folders."""
        for folder in (cls.data_dir, cls.train_dir, cls.val_dir, cls.test_dir, cls.checkpoint_dir):
            folder.mkdir(parents=True, exist_ok=True)
