from io import BytesIO

import numpy as np
import pytest
from PIL import Image

pillow_heif = pytest.importorskip("pillow_heif")
pillow_heif.register_heif_opener()


def _heic_bytes() -> bytes:
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (120, 160, 3), dtype=np.uint8))
    buf = BytesIO()
    pillow_heif.from_pillow(img).save(buf, quality=90)
    return buf.getvalue()


def test_restore_accepts_heic_photo(client):
    """iPhone-style HEIC uploads should be decoded and restored like any other photo."""
    data = _heic_bytes()
    result = client.post("/api/restore", files={"image": ("iphone.heic", data, "image/heic")}, data={"engine": "unet"})
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["width"] > 0 and body["height"] > 0
    img = client.get(body["output_url"])
    assert img.headers["content-type"] == "image/png"
    assert Image.open(BytesIO(img.content)).size == (body["width"], body["height"])
