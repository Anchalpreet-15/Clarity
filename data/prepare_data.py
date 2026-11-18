"""Build the training data: download BSDS500 photos and create degraded/clean pairs.

    python data/prepare_data.py              # download BSDS500, then build the pairs
    python data/prepare_data.py --use-raw    # use your own photos in data/raw_clean/
    python data/prepare_data.py --synthetic  # offline test set of generated images

Output (same layout the training script expects)::

    data/{train,val,test}/clean/00000.png      data/{train,val,test}/degraded/00000.png

Each photo is degraded once: 50% Gaussian noise (sigma 10-50), 25% Gaussian blur
(kernel 5-15, sigma 1-4) and 25% both. The split is 360 train / 40 val / 68 test.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config  # noqa: E402

BSDS500_GIT = "https://github.com/BIDS/BSDS500"
BSDS500_ZIP = "https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

MANUAL_HELP = """
Could not download BSDS500 automatically. Do one of these instead:

  1. Download https://github.com/BIDS/BSDS500 (Code > Download ZIP), copy the .jpg files
     from BSDS500/data/images/{train,val,test}/ into data/raw_clean/, then run:
         python data/prepare_data.py --use-raw

  2. Or generate a small synthetic dataset just to try the pipeline:
         python data/prepare_data.py --synthetic
"""


def _images_in(folder: Path) -> List[Path]:
    """Every image under ``folder``, ordered by file name (keeps the split reproducible)."""
    found = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()]
    return sorted(found, key=lambda p: p.name)


def _download_with_git(workdir: Path) -> Optional[Path]:
    """Shallow-clone the BSDS500 mirror; returns the folder with the photos."""
    if not shutil.which("git"):
        return None
    target = workdir / "bsds500"
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    print("  cloning", BSDS500_GIT)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", BSDS500_GIT, str(target)],
        env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("  ! git clone failed:", result.stderr.strip().splitlines()[-1:] or "unknown error")
        return None
    return target / "BSDS500" / "data" / "images"


def _download_zip(workdir: Path) -> Optional[Path]:
    """Download and unpack the zip archive; returns the folder with the photos."""
    archive = workdir / "bsds500.zip"
    print("  downloading", BSDS500_ZIP)
    try:
        with urllib.request.urlopen(BSDS500_ZIP, timeout=120) as response, open(archive, "wb") as out:
            shutil.copyfileobj(response, out)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(workdir / "bsds500")
    except Exception as exc:  # noqa: BLE001
        print("  ! zip download failed:", exc)
        return None
    return workdir / "bsds500"


def load_bsds500(workdir: Path) -> List[Path]:
    """Fetch BSDS500 with git or as a zip and return all its photo paths."""
    for fetch in (_download_with_git, _download_zip):
        folder = fetch(workdir)
        if folder and folder.exists():
            images = _images_in(folder)
            if len(images) >= 400:
                return images
    return []


def degrade(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply noise (50%), blur (25%) or both (25%) to a uint8 HxWx3 image."""
    roll = rng.random()
    noisy, blurry = roll < 0.5 or roll >= 0.75, roll >= 0.5
    out = img.astype(np.float32)
    if blurry:
        kernel = rng.randrange(Config.blur_kernel_min, Config.blur_kernel_max + 1) | 1
        out = cv2.GaussianBlur(out, (kernel, kernel), rng.uniform(Config.blur_sigma_min, Config.blur_sigma_max))
    if noisy:
        sigma = rng.uniform(Config.noise_sigma_min, Config.noise_sigma_max)
        out += np.random.normal(0.0, sigma, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def synthetic_images(count: int, size: int = 256, seed: int = 0) -> List[Image.Image]:
    """Generate simple shape-based pictures for offline testing."""
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(count):
        canvas = cv2.GaussianBlur(rng.integers(0, 256, (size, size, 3), dtype=np.uint8), (0, 0), 4.0)
        for _ in range(int(rng.integers(3, 8))):
            p1, p2 = (tuple(int(v) for v in rng.integers(0, size, 2)) for _ in range(2))
            cv2.rectangle(canvas, p1, p2, tuple(int(c) for c in rng.integers(0, 256, 3)), -1)
        for _ in range(int(rng.integers(2, 6))):
            centre = tuple(int(v) for v in rng.integers(0, size, 2))
            cv2.circle(canvas, centre, int(rng.integers(5, size // 4)), tuple(int(c) for c in rng.integers(0, 256, 3)), -1)
        images.append(Image.fromarray(canvas))
    return images


def write_pairs(photos: Iterable[Union[Path, Image.Image]], out_dir: Path, rng: random.Random) -> int:
    """Save each photo (opened on demand) and a degraded copy as numbered PNGs."""
    for sub in ("clean", "degraded"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    count = 0
    for item in photos:
        photo = Image.open(item).convert("RGB") if isinstance(item, Path) else item
        name = f"{count:05d}.png"
        photo.save(out_dir / "clean" / name)
        Image.fromarray(degrade(np.asarray(photo), rng)).save(out_dir / "degraded" / name)
        count += 1
    return count


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare restoration training data")
    parser.add_argument("--synthetic", action="store_true", help="generate images instead of downloading")
    parser.add_argument("--use-raw", action="store_true", help="use photos from data/raw_clean/")
    parser.add_argument("--out", type=Path, default=Config.data_dir, help="output folder (default: data/)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    with tempfile.TemporaryDirectory() as tmp:
        if args.synthetic:
            print("Generating synthetic images...")
            photos = synthetic_images(500, seed=args.seed)
        else:
            if args.use_raw:
                paths = _images_in(Config.data_dir / "raw_clean")
            else:
                print("Downloading BSDS500...")
                paths = load_bsds500(Path(tmp))
            if not paths:
                print(MANUAL_HELP)
                return 1
            photos = paths

        random.Random(args.seed).shuffle(photos)
        a, b = Config.train_split, Config.train_split + Config.val_split
        splits = {"train": photos[:a], "val": photos[a:b], "test": photos[b : b + Config.test_split]}
        for name, items in splits.items():
            print(f"  {name}: wrote {write_pairs(items, args.out / name, rng)} pairs to {args.out / name}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
