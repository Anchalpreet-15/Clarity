"""Shrink a training checkpoint into the lean weights file the app ships with.

    python scripts/export_weights.py                          # checkpoints/best_unet.pth -> weights/unet.pth
    python scripts/export_weights.py path/to/checkpoint.pth

Keeps only the model weights, stored in half precision (about 16 MB instead of 94 MB).
Inference converts them back to float32, and the test-set PSNR is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "checkpoints" / "best_unet.pth"
    target = ROOT / "weights" / "unet.pth"
    if not source.exists():
        print(f"Checkpoint not found: {source}")
        return 1
    state = torch.load(source, map_location="cpu", weights_only=True)
    state = state.get("model_state_dict", state)
    lean = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
    target.parent.mkdir(exist_ok=True)
    torch.save(lean, target)
    print(f"Wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
