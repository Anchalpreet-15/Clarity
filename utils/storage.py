"""Accounts, login sessions and per-user restoration history.

Everything lives under ``DATA_DIR`` (default ``./storage``)::

    app.db                         SQLite: users, sessions, restorations
    user_images/<user>/<stamp>_input.png | _output.png | _thumb.jpg

Passwords are stored as salted PBKDF2-SHA256 hashes, session tokens only as
SHA-256 hashes, and each user keeps their most recent ``MAX_HISTORY`` results.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from PIL import Image

from config import PROJECT_ROOT

DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "storage"))
MAX_HISTORY = 30
SESSION_TTL = 30 * 24 * 3600
MIN_PASSWORD = 8

_PBKDF2_ROUNDS = 200_000
_USERNAME_RE = re.compile(r"^[a-z0-9_]{3,32}$")
_KINDS = {"input": "_input.png", "output": "_output.png", "thumb": "_thumb.jpg"}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY, password_hash TEXT NOT NULL,
    salt TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY, expires_at REAL NOT NULL,
    username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS restorations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, stem TEXT NOT NULL, engine TEXT NOT NULL,
    width INTEGER NOT NULL, height INTEGER NOT NULL, psnr REAL, created_at TEXT NOT NULL,
    username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE);
CREATE INDEX IF NOT EXISTS idx_restorations_user ON restorations(username, id DESC);
"""
_ready: set = set()


def db_path() -> Path:
    """Location of the SQLite database."""
    return DATA_DIR / "app.db"


def images_dir() -> Path:
    """Root folder for every user's saved images."""
    return DATA_DIR / "user_images"


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Open the database, creating folders/tables on first use (or after deletion)."""
    path = db_path()
    fresh = path not in _ready or not path.exists()
    if fresh:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        if fresh:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(_SCHEMA)
            _ready.add(path)
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the storage folders and tables if they don't exist yet."""
    images_dir().mkdir(parents=True, exist_ok=True)
    with _connect():
        pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_password(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt_hex), _PBKDF2_ROUNDS
    ).hex()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# --------------------------------------------------------------------- accounts


def create_user(username: str, password: str) -> Tuple[bool, str]:
    """Register an account. Returns ``(ok, message)``; usernames are case-insensitive."""
    username = (username or "").strip().lower()
    if not _USERNAME_RE.match(username):
        return False, "Username must be 3-32 characters: letters, numbers or underscore."
    if len(password or "") < MIN_PASSWORD:
        return False, f"Password must be at least {MIN_PASSWORD} characters."
    salt = secrets.token_hex(16)
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?)",
                (username, _hash_password(password, salt), salt, _now()),
            )
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    return True, username


def verify_user(username: str, password: str) -> Optional[str]:
    """Return the normalised username if the credentials are right, else ``None``."""
    username = (username or "").strip().lower()
    with _connect() as conn:
        row = conn.execute(
            "SELECT password_hash, salt FROM users WHERE username = ?", (username,)
        ).fetchone()
    stored, salt = row if row else ("", secrets.token_hex(16))
    candidate = _hash_password(password or "", salt)
    return username if row and hmac.compare_digest(stored, candidate) else None


def create_session(username: str) -> str:
    """Start a login session and return its secret token."""
    token = secrets.token_urlsafe(32)
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?)",
            (_token_hash(token), time.time() + SESSION_TTL, username),
        )
    return token


def user_for_token(token: str) -> Optional[str]:
    """Username owning a valid, unexpired session token."""
    if not token:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT username FROM sessions WHERE token_hash = ? AND expires_at > ?",
            (_token_hash(token), time.time()),
        ).fetchone()
    return row[0] if row else None


def end_session(token: str) -> None:
    """Invalidate a session token (log out)."""
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))


# ---------------------------------------------------------------------- history


def _remove_files(folder: Path, stem: str) -> None:
    for suffix in _KINDS.values():
        (folder / f"{stem}{suffix}").unlink(missing_ok=True)


def save_restoration(
    username: str,
    engine: str,
    input_img: Image.Image,
    output_img: Image.Image,
    psnr: Optional[float] = None,
) -> int:
    """Store both images plus a thumbnail and return the history id."""
    stem = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    folder = images_dir() / username
    folder.mkdir(parents=True, exist_ok=True)
    input_img.save(folder / f"{stem}_input.png", compress_level=3)
    output_img.save(folder / f"{stem}_output.png", compress_level=3)
    thumb = output_img.copy()
    thumb.thumbnail((360, 360))
    thumb.save(folder / f"{stem}_thumb.jpg", quality=82)

    with _connect() as conn:
        item_id = conn.execute(
            "INSERT INTO restorations (stem, engine, width, height, psnr, created_at, username)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (stem, engine, *output_img.size, psnr, _now(), username),
        ).lastrowid
        stale = conn.execute(
            "SELECT id, stem FROM restorations WHERE username = ? ORDER BY id DESC LIMIT -1 OFFSET ?",
            (username, MAX_HISTORY),
        ).fetchall()
        for old_id, old_stem in stale:
            conn.execute("DELETE FROM restorations WHERE id = ?", (old_id,))
            _remove_files(folder, old_stem)
    return int(item_id)


def list_history(username: str) -> List[dict]:
    """The user's saved restorations, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, engine, width, height, psnr, created_at FROM restorations"
            " WHERE username = ? ORDER BY id DESC",
            (username,),
        ).fetchall()
    keys = ("id", "engine", "width", "height", "psnr", "created_at")
    return [dict(zip(keys, r)) for r in rows]


def get_file(username: str, item_id: int, kind: str) -> Optional[Path]:
    """Path of a saved image (``input``, ``output`` or ``thumb``) if the user owns it."""
    if kind not in _KINDS:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT stem FROM restorations WHERE id = ? AND username = ?", (item_id, username)
        ).fetchone()
    if not row:
        return None
    path = images_dir() / username / f"{row[0]}{_KINDS[kind]}"
    return path if path.exists() else None


def delete_item(username: str, item_id: int) -> bool:
    """Delete one history entry and its files. Returns whether anything was removed."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT stem FROM restorations WHERE id = ? AND username = ?", (item_id, username)
        ).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM restorations WHERE id = ?", (item_id,))
    _remove_files(images_dir() / username, row[0])
    return True
