"""NAFNet (Nonlinear Activation Free Network) deblurring model.

Architecture follows "Simple Baselines for Image Restoration"
(https://arxiv.org/abs/2204.04676) and the reference code at
https://github.com/megvii-research/NAFNet. The GoPro-width32 layout is used:
width 32, encoder blocks (1, 1, 1, 28), 1 middle block, decoder blocks (1, 1, 1, 1).

Weights are fetched on request (``python -m models.nafnet_loader``) from public
Hugging Face mirrors and cached in ``checkpoints/nafnet_weights.pth``. You can also
download ``NAFNet-GoPro-width32.pth`` yourself from the official repo's docs and
place it there. Weights are read with ``weights_only=True`` so no code is executed.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from config import Config
from utils.image_utils import pad_to_multiple, to_pil, to_tensor, unpad


class NAFNetUnavailable(RuntimeError):
    """Raised when NAFNet weights can't be downloaded or loaded."""


# (url, expected sha256 or None). A mirror whose hash differs is skipped.
NAFNET_MIRRORS: List[Tuple[str, Optional[str]]] = [
    (
        "https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-GoPro-width32.pth",
        "19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c",
    ),
    ("https://huggingface.co/mikestealth/nafnet-models/resolve/main/NAFNet-GoPro-width32.pth", None),
]


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an (N, C, H, W) tensor."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """Split channels in two halves and multiply them."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """NAFNet block: SimpleGate + simplified channel attention."""

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2) -> None:
        super().__init__()
        dw = c * dw_expand
        ffn = c * ffn_expand
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.sg = SimpleGate()
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.sg(self.conv2(self.conv1(self.norm1(x))))
        y = self.conv3(y * self.sca(y))
        x = x + y * self.beta
        y = self.conv5(self.sg(self.conv4(self.norm2(x))))
        return x + y * self.gamma


class NAFNet(nn.Module):
    """NAFNet encoder-decoder (defaults = GoPro-width32)."""

    def __init__(
        self,
        img_channel: int = 3,
        width: int = 32,
        middle_blk_num: int = 1,
        enc_blk_nums: Tuple[int, ...] = (1, 1, 1, 28),
        dec_blk_nums: Tuple[int, ...] = (1, 1, 1, 1),
    ) -> None:
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        h, w = inp.shape[-2:]
        x = F.pad(inp, (0, (-w) % self.padder_size, 0, (-h) % self.padder_size))
        skip_in = x
        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for dec, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = dec(up(x) + skip)
        return (self.ending(x) + skip_in)[..., :h, :w]


def weights_path() -> Path:
    """Where the NAFNet weights are cached."""
    return Config.checkpoint_dir / "nafnet_weights.pth"


def download_weights(dest: Optional[Path] = None) -> bool:
    """Download the weights from the first mirror that works. Returns success."""
    dest = dest or weights_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url, sha in NAFNET_MIRRORS:
        part = dest.with_suffix(".part")
        try:
            print(f"[nafnet] downloading {url}")
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    digest.update(chunk)
            if sha and digest.hexdigest() != sha:
                print("[nafnet] checksum mismatch, trying next mirror")
                continue
            part.replace(dest)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[nafnet] failed: {e}")
        finally:
            part.unlink(missing_ok=True)
    return False


_model: Optional[NAFNet] = None


def get_nafnet_model(device: Optional[str] = None) -> NAFNet:
    """Return a cached NAFNet with weights loaded (downloading them if needed)."""
    global _model
    if _model is not None:
        return _model
    path = weights_path()
    if not path.exists() and not download_weights(path):
        raise NAFNetUnavailable(
            "Could not download the NAFNet weights. Save NAFNet-GoPro-width32.pth "
            f"from https://github.com/megvii-research/NAFNet to {path} and retry."
        )
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("params", state)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model = NAFNet()
        model.load_state_dict(state, strict=True)
    except Exception as e:  # noqa: BLE001
        raise NAFNetUnavailable(f"The NAFNet weights at {path} could not be loaded: {e}") from e
    _model = model.eval().to(device or Config.device)
    return _model


@torch.inference_mode()
def restore_with_nafnet(pil_image: Image.Image, device: Optional[str] = None) -> Image.Image:
    """Deblur a PIL image with NAFNet and return a PIL image."""
    device = device or Config.device
    model = get_nafnet_model(device)
    x, pad_h, pad_w = pad_to_multiple(to_tensor(pil_image).to(device), Config.pad_multiple_nafnet)
    return to_pil(unpad(model(x), pad_h, pad_w))


if __name__ == "__main__":
    ok = download_weights()
    print("weights ready" if ok else "download failed")
    sys.exit(0 if ok else 1)
