"""Render a before/after gallery for the docs folder.

    python scripts/make_gallery.py docs/gallery.png

Shows real camera photos and a mix of synthetic damage types, each as a
noisy-input to restored-output to clean-reference row, with PSNR on each tile.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from config import Config  # noqa: E402
from evaluate import CONDITIONS, real_set  # noqa: E402
from models.engine import UNetEngine  # noqa: E402
from utils.metrics import psnr  # noqa: E402

CELL = 320
FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]


def font(size: int) -> ImageFont.ImageFont:
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def zoom(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h, 180)
    box = ((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2)
    return img.crop(box).resize((CELL, CELL), Image.LANCZOS)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "gallery.png"
    engine = UNetEngine(device="cpu")

    rows = []
    real = real_set(Config.data_dir / "real" / "test")
    for i in [0, 2, 5]:
        rows.append(("Real camera photo", *real[i]))
    cleans = sorted((Config.test_dir / "clean").glob("*.png"))
    for cond, idx in [("camera noise (signal-dependent)", 4), ("JPEG quality 35", 20), ("noise sigma 25", 12), ("motion blur 9 px", 32)]:
        clean = Image.open(cleans[idx]).convert("RGB")
        bad = np.clip(CONDITIONS[cond](np.asarray(clean, dtype=np.float32), np.random.default_rng(idx)), 0, 255).astype(np.uint8)
        rows.append((cond, Image.fromarray(bad), clean))

    headers = ["Damaged input", "Restored", "Reference"]
    title_h, label_h, head_h, gap = 36, 44, 60, 12
    row_h = title_h + CELL + label_h + gap
    width = 3 * CELL + 4 * gap
    sheet = Image.new("RGB", (width, head_h + len(rows) * row_h + gap), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 14), "Gallery of before/after examples", fill="#15171c", font=font(24))
    for c, text in enumerate(headers):
        draw.text((gap + c * (CELL + gap) + 6, 42), text, fill="#6b7280", font=font(14))
    for r, (title, noisy, clean) in enumerate(rows):
        y = head_h + r * row_h
        draw.text((gap + 6, y + 6), title, fill="#15171c", font=font(20))
        outputs = [noisy, engine.restore(noisy), clean]
        for c, img in enumerate(outputs):
            x = gap + c * (CELL + gap)
            sheet.paste(zoom(img), (x, y + title_h))
            if c < 2:
                draw.text((x + 6, y + title_h + CELL + 10), f"PSNR vs reference: {psnr(img, clean):.2f} dB", fill="#15171c", font=font(16))
    sheet.save(out)
    print("wrote", out, sheet.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
