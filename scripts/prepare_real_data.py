"""Prepare the PolyU real-world noisy image dataset for training and testing.

    git clone --depth 1 https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset
    python scripts/prepare_real_data.py PolyU-Real-World-Noisy-Images-Dataset

The dataset has 40 full-size scenes (a real noisy shot and the mean of many shots as the
clean reference) plus 100 supplied 512x512 crops. Scenes are split into train / val / test
so no scene appears on both sides: the full-size photos of training scenes become random
training patches, and the supplied crops of the other scenes become the val and test sets.
Each crop is matched to its source scene by pixel content, not by file name.

Output (inside data/real/): train_patches.npy, val/, test/, split.json.
Dataset: Xu et al., "Real-world Noisy Image Denoising: A New Benchmark", 2018.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import Config  # noqa: E402

PATCH = 144


def _small_gray(path: Path, scale: float = 0.25) -> np.ndarray:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def match_crops(dataset: Path, originals: List[str]) -> Dict[str, str]:
    """Find, for every supplied crop, the full-size scene it was cut from."""
    small = {name: _small_gray(dataset / "OriginalImages" / f"{name}_mean.JPG") for name in originals}
    mapping = {}
    for crop in sorted((dataset / "CroppedImages").glob("*_mean.JPG")):
        template = _small_gray(crop)
        scores = {n: cv2.minMaxLoc(cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED))[1] for n, img in small.items()}
        best = max(scores, key=scores.get)
        if scores[best] < 0.85:
            raise RuntimeError(f"Could not match {crop.name} to a scene (best score {scores[best]:.2f})")
        mapping[crop.name.replace("_mean.JPG", "")] = best
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare real-noise data")
    parser.add_argument("dataset", type=Path, help="folder of the cloned PolyU dataset")
    parser.add_argument("--patches-per-scene", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    originals = sorted(p.name.replace("_Real.JPG", "") for p in (args.dataset / "OriginalImages").glob("*_Real.JPG"))
    if not originals:
        print("No OriginalImages/*_Real.JPG found; is this the PolyU dataset folder?")
        return 1
    split = {"test": originals[0::4], "val": originals[2::8]}
    split["train"] = [n for n in originals if n not in split["test"] + split["val"]]
    print({k: len(v) for k, v in split.items()}, "scenes")

    print("Matching crops to their source scenes by pixel content...")
    crop_scene = match_crops(args.dataset, originals)
    out_dir = Config.data_dir / "real"
    for part in ("val", "test"):
        folder = out_dir / part
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        crops = [c for c, scene in crop_scene.items() if scene in split[part]]
        for crop in crops:
            for suffix in ("real", "mean"):
                shutil.copy(args.dataset / "CroppedImages" / f"{crop}_{suffix}.JPG", folder)
        print(f"  {part}: {len(crops)} crops from {len(split[part])} scenes")

    rng = np.random.default_rng(args.seed)
    total = len(split["train"]) * args.patches_per_scene
    patches = np.lib.format.open_memmap(out_dir / "train_patches.npy", mode="w+", dtype=np.uint8, shape=(total, 2, PATCH, PATCH, 3))
    i = 0
    for scene in split["train"]:
        noisy = cv2.cvtColor(cv2.imread(str(args.dataset / "OriginalImages" / f"{scene}_Real.JPG")), cv2.COLOR_BGR2RGB)
        clean = cv2.cvtColor(cv2.imread(str(args.dataset / "OriginalImages" / f"{scene}_mean.JPG")), cv2.COLOR_BGR2RGB)
        h, w = noisy.shape[:2]
        for _ in range(args.patches_per_scene):
            top, left = int(rng.integers(0, h - PATCH)), int(rng.integers(0, w - PATCH))
            patches[i, 0] = noisy[top : top + PATCH, left : left + PATCH]
            patches[i, 1] = clean[top : top + PATCH, left : left + PATCH]
            i += 1
    patches.flush()
    (out_dir / "split.json").write_text(json.dumps({**split, "crop_scene": crop_scene}, indent=1))
    print(f"  train: {total} patches of {PATCH}x{PATCH} from {len(split['train'])} scenes -> {out_dir / 'train_patches.npy'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
