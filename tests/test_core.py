import io

import numpy as np
import pytest
import torch
from PIL import Image

from models import engine
from models.nafnet_loader import NAFNet
from models.unet import UNet
from utils.image_utils import pad_to_multiple, to_pil, to_tensor, unpad
from utils.metrics import psnr, ssim


@pytest.mark.parametrize("h,w", [(16, 16), (33, 47), (128, 96)])
def test_unet_keeps_size(h, w):
    net = UNet(base=8).eval()
    with torch.no_grad():
        assert net(torch.rand(1, 3, h, w)).shape == (1, 3, h, w)


def test_pad_roundtrip_and_tiny_images():
    for h, w in [(5, 7), (40, 40), (17, 33)]:
        x = torch.rand(1, 3, h, w)
        padded, ph, pw = pad_to_multiple(x, 16)
        assert padded.shape[-1] % 16 == 0 and padded.shape[-2] % 16 == 0
        assert torch.equal(unpad(padded, ph, pw), x)


def test_tensor_pil_roundtrip():
    img = Image.fromarray(np.random.randint(0, 256, (20, 30, 3), dtype=np.uint8))
    assert np.array_equal(np.array(to_pil(to_tensor(img))), np.array(img))


def test_metrics():
    a = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
    assert psnr(a, a) == 99.0 and ssim(a, a) == pytest.approx(1.0)
    assert psnr(a, a.point(lambda v: min(255, v + 20))) < 40


def test_tiled_matches_whole_image(monkeypatch, sample_png):
    eng = engine.UNetEngine(device="cpu")
    crop = Image.open(io.BytesIO(sample_png)).convert("RGB").crop((0, 0, 300, 220))
    x = to_tensor(crop).contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        whole = eng.model(x)
    monkeypatch.setattr(engine, "WHOLE_IMAGE_PIXELS", 1000)
    tiled = engine.tiled_forward(eng.model, x, tile=96, margin=48)
    assert psnr(tiled, whole) > 45


def test_nafnet_gopro_layout_has_official_size():
    assert sum(p.numel() for p in NAFNet().parameters()) == pytest.approx(17.11e6, rel=0.001)


def test_unet_engine_restores_with_shipped_weights():
    out = engine.UNetEngine(device="cpu").restore(Image.new("RGB", (70, 50), "gray"))
    assert out.size == (70, 50)
