"""Render a side-by-side comparison of several weight files on the same photos.

    python scripts/compare_models.py --weights runs/v1/unet.pth runs/v2/best_unet.pth \
        --names "Before" "After" --real data/real/test --out comparison.png

Each row shows the damaged input, one column per model and the clean reference, zoomed in
on a patch so the differences are visible. Rows are real camera-noise photos (if --real is
given) followed by synthetic damage types. PSNR is for the whole image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from config import Config  # noqa: E402
from evaluate import CONDITIONS, real_set  # noqa: E402
from models.engine import UNetEngine  # noqa: E402
from utils.metrics import psnr  # noqa: E402

CELL = 360
FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]


def font(size: int) -> ImageFont.ImageFont:
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def zoom(img: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    return img.crop(box).resize((CELL, CELL), Image.LANCZOS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Side-by-side model comparison")
    parser.add_argument("--weights", nargs="+", type=Path, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--real", type=Path, help="folder of real *_real.JPG / *_mean.JPG pairs")
    parser.add_argument("--real-count", type=int, default=3)
    parser.add_argument("--synthetic", nargs="*", default=["camera noise (signal-dependent)", "motion blur 9 px", "noise sigma 25"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.weights) != len(args.names):
        parser.error("--names must match --weights")

    engines = [UNetEngine(device="cpu", weights=w) for w in args.weights]
    rows: List[Tuple[str, Image.Image, Image.Image]] = []
    if args.real:
        pairs = real_set(args.real)
        for i in np.linspace(0, len(pairs) - 1, args.real_count).astype(int):
            rows.append(("Real camera noise", *pairs[i]))
    clean_files = sorted((Config.test_dir / "clean").glob("*.png"))
    for k, name in enumerate(args.synthetic):
        clean = Image.open(clean_files[5 + 7 * k]).convert("RGB")
        bad = np.clip(CONDITIONS[name](np.asarray(clean, dtype=np.float32), np.random.default_rng(k + 1)), 0, 255).astype(np.uint8)
        rows.append((name, Image.fromarray(bad), clean))

    headers = ["Input", *args.names, "Reference"]
    title_h, label_h, head_h, gap = 36, 40, 56, 8
    row_h = title_h + CELL + label_h + gap
    width = len(headers) * CELL + (len(headers) + 1) * gap
    sheet = Image.new("RGB", (width, head_h + len(rows) * row_h + gap), "white")
    draw = ImageDraw.Draw(sheet)
    for c, text in enumerate(headers):
        draw.text((gap + c * (CELL + gap) + 6, 14), text, fill="#15171c", font=font(26))
    for r, (title, noisy, clean) in enumerate(rows):
        w, h = noisy.size
        size = min(w, h, 200)
        box = ((w - size) // 2, (h - size) // 2, (w + size) // 2, (h + size) // 2)
        images = [noisy, *[e.restore(noisy) for e in engines], clean]
        y = head_h + r * row_h
        draw.text((gap + 6, y + 6), title, fill="#15171c", font=font(22))
        for c, img in enumerate(images):
            x = gap + c * (CELL + gap)
            sheet.paste(zoom(img, box), (x, y + title_h))
            if c < len(images) - 1:
                draw.text((x + 6, y + title_h + CELL + 8), f"{psnr(img, clean):.2f} dB", fill="#15171c", font=font(21))
    sheet.save(args.out)
    print("wrote", args.out, sheet.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
