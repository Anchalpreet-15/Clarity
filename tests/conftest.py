import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["NAFNET_AUTODOWNLOAD"] = "0"
os.environ["NO_BROWSER"] = "1"

import pytest  # noqa: E402

from utils import storage  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A test client whose accounts and images live in a throw-away folder."""
    from fastapi.testclient import TestClient

    import app as app_module

    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    storage._ready.clear()
    for limiter in (app_module.auth_limiter, app_module.signup_limiter, app_module.restore_limiter):
        limiter._hits.clear()
    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture(scope="session")
def sample_png() -> bytes:
    return (ROOT / "web" / "samples" / "light.png").read_bytes()


@pytest.fixture(scope="session")
def sample_clean_png() -> bytes:
    return (ROOT / "web" / "samples" / "light-clean.png").read_bytes()
