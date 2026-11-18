"""Benchmark one or more weight files on synthetic and real-noise test sets.

    python scripts/evaluate.py --weights weights/unet.pth runs/v2/best_unet.pth
    python scripts/evaluate.py --weights weights/unet.pth --real path/to/PolyU/CroppedImages

Synthetic conditions use the held-out photos in data/test/clean with fixed random seeds,
so numbers are comparable between runs. ``--real`` adds real camera-noise pairs
(files named ``*_real.JPG`` with a matching ``*_mean.JPG`` reference).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import Config  # noqa: E402
from data.degradations import disc_kernel, jpeg_roundtrip, motion_kernel  # noqa: E402
from models.engine import UNetEngine  # noqa: E402
from utils.metrics import psnr, ssim  # noqa: E402


def _blur(sigma: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    kernel = 2 * int(np.ceil(3 * sigma)) + 1
    return lambda a, rng: cv2.GaussianBlur(a, (kernel, kernel), sigma)


def _noise(sigma: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    return lambda a, rng: a + rng.normal(0, sigma, a.shape).astype(np.float32)


def _kernel(kernel: np.ndarray) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    return lambda a, rng: cv2.filter2D(a, -1, kernel, borderType=cv2.BORDER_REFLECT)


def _camera_noise(a: float, b: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    """Signal-dependent (Poisson-Gaussian) noise, like a camera sensor at high ISO."""
    return lambda x, rng: x + rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(np.clip(x / 255.0, 0, 1) * a + b) * 255.0


def _jpeg(quality: int) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    return lambda a, rng: jpeg_roundtrip(np.clip(a, 0, 255).astype(np.uint8), quality).astype(np.float32)


CONDITIONS: Dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "noise sigma 3 (barely visible)": _noise(3),
    "noise sigma 10": _noise(10),
    "noise sigma 25": _noise(25),
    "noise sigma 50": _noise(50),
    "blur sigma 1.0": _blur(1.0),
    "blur sigma 2.0": _blur(2.0),
    "blur sigma 3.0": _blur(3.0),
    "camera noise (signal-dependent)": _camera_noise(0.004, 2e-5),
    "motion blur 9 px": _kernel(motion_kernel(9, 0.4)),
    "defocus blur radius 3": _kernel(disc_kernel(3.0)),
    "JPEG quality 35": _jpeg(35),
    "blur 1.5 + noise 20": lambda a, rng: _noise(20)(_blur(1.5)(a, rng), rng),
    "blur 2.5 + noise 10": lambda a, rng: _noise(10)(_blur(2.5)(a, rng), rng),
}


def score(engine: UNetEngine, pairs: List[Tuple[Image.Image, Image.Image]]) -> Dict[str, float]:
    """Mean PSNR/SSIM of the input and of the restored image against the reference."""
    totals = {"in_psnr": 0.0, "out_psnr": 0.0, "in_ssim": 0.0, "out_ssim": 0.0}
    for degraded, clean in pairs:
        restored = engine.restore(degraded)
        totals["in_psnr"] += psnr(degraded, clean)
        totals["out_psnr"] += psnr(restored, clean)
        totals["in_ssim"] += ssim(degraded, clean)
        totals["out_ssim"] += ssim(restored, clean)
    return {k: v / len(pairs) for k, v in totals.items()}


def synthetic_sets(limit: int) -> Dict[str, List[Tuple[Image.Image, Image.Image]]]:
    """Degrade the held-out test photos once per condition (fixed seeds)."""
    cleans = [Image.open(p).convert("RGB") for p in sorted((Config.test_dir / "clean").glob("*.png"))[:limit]]
    sets = {}
    for name, fn in CONDITIONS.items():
        rng = np.random.default_rng(sum(map(ord, name)))
        pairs = []
        for clean in cleans:
            bad = np.clip(fn(np.asarray(clean, dtype=np.float32), rng), 0, 255).astype(np.uint8)
            pairs.append((Image.fromarray(bad), clean))
        sets[name] = pairs
    blind = [(Image.open(p).convert("RGB"), Image.open(str(p).replace("degraded", "clean")).convert("RGB"))
             for p in sorted((Config.test_dir / "degraded").glob("*.png"))[:limit]]
    return {"mixed (original test set)": blind, **sets}


def real_set(folder: Path) -> List[Tuple[Image.Image, Image.Image]]:
    """Real camera-noise pairs: ``*_real.JPG`` against its ``*_mean.JPG`` reference."""
    pairs = []
    for noisy in sorted(folder.glob("*_real.JPG")):
        mean = noisy.with_name(noisy.name.replace("_real", "_mean"))
        if mean.exists():
            pairs.append((Image.open(noisy).convert("RGB"), Image.open(mean).convert("RGB")))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark restoration weights")
    parser.add_argument("--weights", nargs="+", type=Path, required=True)
    parser.add_argument("--real", type=Path, help="folder with *_real.JPG / *_mean.JPG pairs")
    parser.add_argument("--limit", type=int, default=68, help="synthetic photos per condition")
    parser.add_argument("--json", type=Path, help="write the results here")
    args = parser.parse_args()

    sets = synthetic_sets(args.limit)
    if args.real:
        sets["REAL camera noise (PolyU)"] = real_set(args.real)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for weights in args.weights:
        engine = UNetEngine(device="cpu", weights=weights)
        results[str(weights)] = {name: score(engine, pairs) for name, pairs in sets.items()}

    names = list(args.weights)
    width = max(len(n) for n in sets) + 2
    print(f"{'condition':<{width}} {'input':>8} " + " ".join(f"{n.parent.name + '/' + n.name:>26}" for n in names))
    for name in sets:
        base = results[str(names[0])][name]["in_psnr"]
        cells = []
        for n in names:
            r = results[str(n)][name]
            cells.append(f"{r['out_psnr']:6.2f} dB ({r['out_psnr'] - base:+5.2f}) {r['out_ssim']:.3f}")
        print(f"{name:<{width}} {base:7.2f}  " + " ".join(f"{c:>26}" for c in cells))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
