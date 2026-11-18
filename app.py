"""Clarity - AI photo restoration web app.

Run ``python app.py`` and open http://localhost:7860.

The browser UI lives in ``web/``; this file serves it together with a small JSON
API (restore a photo, optional accounts, per-user history). Photos from visitors
who are not signed in are processed in memory and never written to disk.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import secrets
import threading
import time
import webbrowser
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager
from typing import Deque, Dict, Optional

import torch
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

try:  # iPhone and some Android photos are HEIC; stay optional for lean installs
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover - optional dependency
    pass

from config import PROJECT_ROOT, Config
from models.engine import NAFNetEngine, load_engines
from utils import storage
from utils.metrics import psnr as compute_psnr
from utils.metrics import ssim as compute_ssim

log = logging.getLogger("clarity")

WEB_DIR = PROJECT_ROOT / "web"
MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", 12)) * 1024 * 1024)
MAX_SIDE = int(os.environ.get("MAX_SIDE", 1600))
MAX_PIXELS = 40_000_000
MIN_SIDE = 32
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", 6))
RESULT_TTL = 15 * 60
RESULT_CACHE_SIZE = 24

CSP = (
    "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; "
    "script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'"
)


class RateLimiter:
    """Sliding-window limiter keyed by client address."""

    def __init__(self, limit: int, window: float) -> None:
        self.limit, self.window = limit, window
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Record a hit for ``key``; return False once it exceeds the limit."""
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > 10_000:
                self._hits.clear()
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True


class ResultCache:
    """Small in-memory store so guests can fetch and download a result once."""

    def __init__(self, size: int, ttl: float) -> None:
        self.size, self.ttl = size, ttl
        self._items: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._items[key] = {**value, "at": time.monotonic()}
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            item = self._items.get(key)
            if item and time.monotonic() - item["at"] > self.ttl:
                del self._items[key]
                return None
            return item


results = ResultCache(RESULT_CACHE_SIZE, RESULT_TTL)
auth_limiter = RateLimiter(10, 300)
signup_limiter = RateLimiter(5, 3600)
restore_limiter = RateLimiter(20, 600)


def _fetch_nafnet_in_background(app: FastAPI) -> None:
    """Try to download the optional NAFNet weights without blocking start-up."""
    from models.nafnet_loader import download_weights, weights_path

    if not weights_path().exists() and not download_weights():
        log.info("NAFNet weights unavailable; continuing with the standard model only.")
        return
    if NAFNetEngine.available():
        app.state.engines["nafnet"] = NAFNetEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models and storage before serving the first request."""
    storage.init_db()
    app.state.engines = await run_in_threadpool(load_engines)
    app.state.gate = asyncio.Semaphore(1)
    app.state.pending = 0
    await run_in_threadpool(app.state.engines["unet"].restore, Image.new("RGB", (64, 64)))
    if os.environ.get("NAFNET_AUTODOWNLOAD", "1") == "1" and "nafnet" not in app.state.engines:
        threading.Thread(target=_fetch_nafnet_in_background, args=(app,), daemon=True).start()
    yield


app = FastAPI(title="Clarity", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.middleware("http")
async def guard(request: Request, call_next):
    """Reject oversized uploads early and add security headers."""
    if request.url.path == "/api/restore":
        declared = int(request.headers.get("content-length") or 0)
        if declared > MAX_UPLOAD_BYTES * 2 + 1024:
            return JSONResponse({"detail": "That upload is too large."}, status_code=413)
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.exception_handler(Exception)
async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    """Never leak a traceback to the browser."""
    log.exception("Unhandled error", exc_info=exc)
    return JSONResponse({"detail": "Something went wrong on our side. Please try again."}, status_code=500)


# ------------------------------------------------------------------------ helpers


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or (request.client.host if request.client else "unknown")


def current_user(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Username for a valid ``Authorization: Bearer <token>`` header, else None."""
    if authorization and authorization.lower().startswith("bearer "):
        return storage.user_for_token(authorization[7:].strip())
    return None


def require_user(user: Optional[str] = Depends(current_user)) -> str:
    if not user:
        raise HTTPException(401, "Please sign in first.")
    return user


def decode_image(data: bytes, label: str = "photo") -> Image.Image:
    """Open an upload safely and return an upright RGB image."""
    try:
        img = Image.open(io.BytesIO(data))
        width, height = img.size
        if width * height > MAX_PIXELS:
            raise HTTPException(413, f"That {label} is too large ({width}x{height}). Please use one under 40 megapixels.")
        img.load()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, f"That {label} couldn't be opened. Please use a JPG, PNG, WebP or HEIC image.")
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        img = Image.new("RGB", rgba.size, "white")
        img.paste(rgba, mask=rgba.getchannel("A"))
    else:
        img = img.convert("RGB")
    if min(img.size) < MIN_SIDE:
        raise HTTPException(400, f"That {label} is too small. Both sides must be at least {MIN_SIDE} pixels.")
    return img


def fit_to_limit(img: Image.Image) -> tuple[Image.Image, bool]:
    """Downscale so the long side is at most MAX_SIDE. Returns (image, was_resized)."""
    longest = max(img.size)
    if longest <= MAX_SIDE:
        return img, False
    scale = MAX_SIDE / longest
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS), True


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=3)
    return buf.getvalue()


def safe_name(filename: Optional[str]) -> str:
    stem = os.path.splitext(os.path.basename(filename or "photo"))[0]
    return (re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.")[:60] or "photo") + "-restored.png"


async def read_upload(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"That file is larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB.")
    return data


def run_restore(engine, original: Image.Image, reference: Optional[Image.Image],
                resized: bool, filename: Optional[str], user: Optional[str]) -> dict:
    """Restore the photo, score it against the reference if given, and store the result."""
    started = time.perf_counter()
    output = engine.restore(original)
    seconds = time.perf_counter() - started

    metrics, note = None, None
    if reference is not None:
        if abs(reference.width / reference.height - original.width / original.height) > 0.02:
            note = "The reference photo has a different shape, so quality scores were skipped."
        else:
            ref = reference if reference.size == output.size else reference.resize(output.size, Image.LANCZOS)
            metrics = {
                "psnr": compute_psnr(output, ref), "ssim": compute_ssim(output, ref),
                "input_psnr": compute_psnr(original, ref), "input_ssim": compute_ssim(original, ref),
            }

    history_id = None
    if user:
        history_id = storage.save_restoration(user, engine.id, original, output, metrics and metrics["psnr"])

    rid = secrets.token_urlsafe(12)
    results.put(rid, {"output": png_bytes(output), "input": png_bytes(original) if resized else None,
                      "name": safe_name(filename)})
    return {
        "id": rid, "engine": engine.id, "width": output.width, "height": output.height,
        "seconds": round(seconds, 1), "resized": resized, "metrics": metrics, "note": note,
        "history_id": history_id, "output_url": f"/api/result/{rid}",
        "input_url": f"/api/result/{rid}?kind=input" if resized else None,
        "download_url": f"/api/result/{rid}?download=1",
    }


# -------------------------------------------------------------------------- API


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/api/status")
def status(request: Request) -> dict:
    engines = request.app.state.engines
    return {
        "engines": [{"id": e.id, "label": e.label, "description": e.description} for e in engines.values()],
        "limits": {"max_upload_mb": MAX_UPLOAD_BYTES // 1024 // 1024, "max_side": MAX_SIDE},
        "device": Config.device,
    }


@app.post("/api/restore")
async def restore(
    request: Request,
    image: UploadFile = File(...),
    reference: Optional[UploadFile] = File(None),
    engine: str = Form("unet"),
    user: Optional[str] = Depends(current_user),
) -> dict:
    engines = request.app.state.engines
    if engine not in engines:
        raise HTTPException(400, "That restoration mode isn't available.")
    if not restore_limiter.allow(client_ip(request)):
        raise HTTPException(429, "You've restored a lot of photos in a short time. Please wait a few minutes.")

    data = await read_upload(image)
    ref_data = await read_upload(reference) if reference is not None and reference.filename else None
    original, resized = await run_in_threadpool(lambda: fit_to_limit(decode_image(data)))
    ref_img = await run_in_threadpool(decode_image, ref_data, "reference photo") if ref_data else None

    state = request.app.state
    if state.pending >= MAX_QUEUE:
        raise HTTPException(503, "We're busy right now. Please try again in a minute.")
    state.pending += 1
    try:
        async with state.gate:
            return await run_in_threadpool(
                run_restore, engines[engine], original, ref_img, resized, image.filename, user
            )
    finally:
        state.pending -= 1


@app.get("/api/result/{rid}")
def result(rid: str, kind: str = "output", download: int = 0) -> Response:
    item = results.get(rid) if re.fullmatch(r"[A-Za-z0-9_-]{8,40}", rid) else None
    data = item and item.get("input" if kind == "input" else "output")
    if not data:
        raise HTTPException(404, "That result has expired. Please restore the photo again.")
    headers = {"Cache-Control": "private, max-age=900"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{item["name"]}"'
    return Response(data, media_type="image/png", headers=headers)


class Credentials(BaseModel):
    username: str
    password: str


@app.post("/api/auth/signup")
def signup(body: Credentials, request: Request) -> dict:
    if not signup_limiter.allow(client_ip(request)):
        raise HTTPException(429, "Too many sign-ups from this network. Please try again later.")
    ok, detail = storage.create_user(body.username, body.password)
    if not ok:
        raise HTTPException(400, detail)
    return {"username": detail, "token": storage.create_session(detail)}


@app.post("/api/auth/login")
def login(body: Credentials, request: Request) -> dict:
    if not auth_limiter.allow(client_ip(request)):
        raise HTTPException(429, "Too many attempts. Please wait a few minutes and try again.")
    username = storage.verify_user(body.username, body.password)
    if not username:
        raise HTTPException(401, "Wrong username or password.")
    return {"username": username, "token": storage.create_session(username)}


@app.post("/api/auth/logout")
def logout(authorization: Optional[str] = Header(None)) -> dict:
    if authorization and authorization.lower().startswith("bearer "):
        storage.end_session(authorization[7:].strip())
    return {"ok": True}


@app.get("/api/me")
def me(user: str = Depends(require_user)) -> dict:
    return {"username": user}


@app.get("/api/history")
def history(user: str = Depends(require_user)) -> dict:
    return {"items": storage.list_history(user)}


@app.get("/api/history/{item_id}/{kind}")
def history_file(item_id: int, kind: str, user: str = Depends(require_user)) -> FileResponse:
    path = storage.get_file(user, item_id, kind)
    if not path:
        raise HTTPException(404, "That image no longer exists.")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})


@app.delete("/api/history/{item_id}")
def history_delete(item_id: int, user: str = Depends(require_user)) -> dict:
    if not storage.delete_item(user, item_id):
        raise HTTPException(404, "That image no longer exists.")
    return {"ok": True}


# ------------------------------------------------------------------- web pages

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})


def main() -> None:
    """Start the web server (host/port come from the HOST and PORT variables)."""
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 7860))
    if host in ("127.0.0.1", "localhost") and not os.environ.get("NO_BROWSER"):
        threading.Timer(2.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    print(f"\n  Clarity is starting on http://localhost:{port}  (press Ctrl+C to stop)\n")
    # Honour TORCH_NUM_THREADS when set (free-tier containers report host cores, not the slice they get).
    requested = int(os.environ.get("TORCH_NUM_THREADS") or os.cpu_count() or 1)
    torch.set_num_threads(max(1, min(requested, os.cpu_count() or requested)))
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
