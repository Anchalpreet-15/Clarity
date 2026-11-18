"""Publish the app as a free Hugging Face Space (Docker).

    pip install huggingface_hub
    huggingface-cli login                       # token with "write" access, or set HF_TOKEN
    python scripts/deploy_to_huggingface.py YOUR_USERNAME/clarity

Add --dry-run to list exactly which files would be uploaded without contacting Hugging Face.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Only what the running app needs: no training code, datasets or checkpoints.
INCLUDE = [
    "app.py", "config.py", "Dockerfile", "requirements.txt",
    "models/*.py", "utils/*.py", "web/*", "web/samples/*", "weights/*.pth",
]

SPACE_CARD = """---
title: Clarity
emoji: C
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: A residual U-Net for photograph denoising
---

# Clarity

A residual U-Net for photograph denoising, trained from scratch in PyTorch.
Source code: https://github.com/Anchalpreet-15/VAE
"""


def files_to_upload() -> list[Path]:
    """Files under the project root that match INCLUDE (excluding caches)."""
    found = set()
    for pattern in INCLUDE:
        found.update(p for p in ROOT.glob(pattern) if p.is_file() and "__pycache__" not in p.parts)
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy to a Hugging Face Space")
    parser.add_argument("repo_id", help="your-username/space-name, e.g. alex/clarity")
    parser.add_argument("--dry-run", action="store_true", help="only list the files")
    args = parser.parse_args()

    files = files_to_upload()
    total = sum(p.stat().st_size for p in files) / 1e6
    print(f"{len(files)} files, {total:.1f} MB to upload:")
    for path in files:
        print("  ", path.relative_to(ROOT))
    if args.dry_run:
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Install the client first:  pip install huggingface_hub")
        return 1

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    try:
        api.whoami()
    except Exception:  # noqa: BLE001
        print("Not logged in. Run `huggingface-cli login` (or set HF_TOKEN) and try again.")
        return 1

    api.create_repo(args.repo_id, repo_type="space", space_sdk="docker", exist_ok=True, token=token)
    api.upload_file(
        path_or_fileobj=SPACE_CARD.encode(), path_in_repo="README.md",
        repo_id=args.repo_id, repo_type="space", commit_message="Add Space card", token=token,
    )
    api.upload_folder(
        folder_path=ROOT, repo_id=args.repo_id, repo_type="space",
        allow_patterns=INCLUDE, commit_message="Deploy Clarity", token=token,
    )
    owner, name = args.repo_id.split("/", 1)
    print(f"\nUploaded. The build takes a few minutes; watch it at https://huggingface.co/spaces/{args.repo_id}")
    print(f"Your app will be live at https://{owner.lower()}-{name.lower()}.hf.space")
    return 0


if __name__ == "__main__":
    sys.exit(main())
