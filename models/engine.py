"""Inference engines: a trained network behind ``restore(PIL) -> PIL``.

The U-Net engine loads ``checkpoints/best_unet.pth`` if you trained your own
model, otherwise the lean weights shipped in ``weights/unet.pth``. Large images
are processed in overlapping tiles so memory stays flat, and on CPUs with fast
bfloat16 support (checked with a tiny timing test at startup) that precision is
used for roughly a 3x speed-up.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image

from config import PROJECT_ROOT, Config
from models.unet import UNet
from utils.image_utils import to_pil, to_tensor

SHIPPED_WEIGHTS = PROJECT_ROOT / "weights" / "unet.pth"
TILE = 512
MARGIN = 48
WHOLE_IMAGE_PIXELS = 640 * 640


def load_unet_state(weights: Optional[Path] = None) -> Dict[str, torch.Tensor]:
    """Return float32 U-Net weights from ``weights``, your own checkpoint or the shipped file."""
    candidates = [Path(weights)] if weights else [Config.checkpoint_dir / "best_unet.pth", SHIPPED_WEIGHTS]
    for path in candidates:
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=True)
            state = state.get("model_state_dict", state)
            return {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
    raise FileNotFoundError(
        "No U-Net weights found. Expected weights/unet.pth (shipped with the project) "
        "or checkpoints/best_unet.pth (created by `python train.py`)."
    )


@torch.inference_mode()
def tiled_forward(model: torch.nn.Module, x: torch.Tensor, tile: int = TILE, margin: int = MARGIN) -> torch.Tensor:
    """Run ``model`` on ``x`` in tiles with ``margin`` pixels of context around each."""
    h, w = x.shape[-2:]
    if h * w <= WHOLE_IMAGE_PIXELS:
        return model(x)
    out = torch.empty_like(x)
    for top in range(0, h, tile):
        for left in range(0, w, tile):
            bottom, right = min(top + tile, h), min(left + tile, w)
            t0, l0 = max(top - margin, 0), max(left - margin, 0)
            t1, l1 = min(bottom + margin, h), min(right + margin, w)
            y = model(x[..., t0:t1, l0:l1])
            out[..., top:bottom, left:right] = y[..., top - t0 : bottom - t0, left - l0 : right - l0]
    return out


class UNetEngine:
    """The project's own residual U-Net."""

    id = "unet"
    label = "Standard"
    description = "Residual U-Net trained from scratch. Best all-rounder for grain and mild blur."

    def __init__(self, device: Optional[str] = None, weights: Optional[Path] = None) -> None:
        self.device = device or Config.device
        model = UNet(base=Config.base_channels)
        model.load_state_dict(load_unet_state(weights))
        self.model = model.eval().to(self.device, memory_format=torch.channels_last)
        self.bf16 = self._choose_precision()

    def _timed(self, bf16: bool) -> float:
        probe = torch.rand(1, 3, 192, 192).to(self.device).contiguous(memory_format=torch.channels_last)
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                self.model(probe)
            best = min(best, time.perf_counter() - start)
        return best

    @torch.inference_mode()
    def _choose_precision(self) -> bool:
        """Use bfloat16 on CPU only when a quick benchmark shows a clear win."""
        setting = os.environ.get("RESTORE_PRECISION", "auto").lower()
        if self.device != "cpu" or setting == "fp32":
            return False
        if setting == "bf16":
            return True
        try:
            return self._timed(True) < 0.7 * self._timed(False)
        except Exception:  # noqa: BLE001
            return False

    @torch.inference_mode()
    def restore(self, image: Image.Image) -> Image.Image:
        """Return the restored version of an RGB image."""
        x = to_tensor(image).to(self.device).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=self.bf16):
            y = tiled_forward(self.model, x)
        return to_pil(y)


class NAFNetEngine:
    """Optional pretrained NAFNet deblurring model (needs its weights file)."""

    id = "nafnet"
    label = "Deblur (NAFNet)"
    description = "Pretrained NAFNet-GoPro. Stronger on motion blur, does not remove grain."

    @staticmethod
    def available() -> bool:
        from models.nafnet_loader import weights_path

        return weights_path().exists()

    def restore(self, image: Image.Image) -> Image.Image:
        """Return the deblurred version of an RGB image."""
        from models.nafnet_loader import restore_with_nafnet

        return restore_with_nafnet(image)


class EnhanceEngine:
    """Optional Real-ESRGAN super-resolution that adds sharpness and detail.

    Unlike the denoiser, this upscales 4x and reconstructs texture, so soft or
    small photos come out visibly crisper. It is heavy on a CPU (seconds to tens
    of seconds per photo), so it is loaded only when ``ENABLE_ENHANCE=1`` is set
    and never on the constrained free hosting tiers.
    """

    id = "enhance"
    label = "Enhance (sharpen + upscale)"
    description = "Real-ESRGAN super-resolution: adds sharpness and detail and upscales 4x. Much slower; best for small or soft photos."

    @staticmethod
    def available() -> bool:
        from models.esrgan import weights_path

        return weights_path().exists()

    def __init__(self) -> None:
        from models.esrgan import ESRGANModel

        self._model = ESRGANModel(max_in=int(os.environ.get("ENHANCE_MAX_IN", 768)))

    def restore(self, image: Image.Image) -> Image.Image:
        return self._model.enhance(image)


def load_engines() -> Dict[str, object]:
    """Instantiate every engine that can run right now, keyed by id."""
    engines: Dict[str, object] = {"unet": UNetEngine()}
    if NAFNetEngine.available():
        engines["nafnet"] = NAFNetEngine()
    if os.environ.get("ENABLE_ENHANCE", "0") == "1" and EnhanceEngine.available():
        engines["enhance"] = EnhanceEngine()
    return engines
