"""Train the residual U-Net.

Training photos are degraded freshly every time they are drawn (see data/degradations.py),
optionally mixed with real camera-noise pairs. Validation uses fixed synthetic pairs and, if
available, held-out real pairs; the best checkpoint maximises the sum of the PSNR gains.

    python train.py                                   # resume or start from scratch
    python train.py --init weights/unet.pth           # fine-tune an existing model
    python train.py --iters-per-epoch 20 --epochs 1 --checkpoint-dir /tmp/ck   # smoke test

Writes last_unet.pth (resume), best_unet.pth (used by the app) and history.csv.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config  # noqa: E402
from data.dataset import MixedDataset, OnTheFlyDataset, RealPairDataset, RealPatchDataset, RestorationDataset  # noqa: E402
from models.engine import load_unet_state  # noqa: E402
from models.unet import UNet  # noqa: E402
from utils.metrics import psnr  # noqa: E402

CHANNELS_LAST = torch.channels_last


class RestorationLoss(nn.Module):
    """L1 loss plus an optional edge (Sobel gradient) term that rewards sharp results."""

    def __init__(self, grad_weight: float = 0.0) -> None:
        super().__init__()
        self.grad_weight = grad_weight
        sobel = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        kernel = torch.stack([sobel, sobel.t()]).unsqueeze(1).repeat(3, 1, 1, 1)  # (6, 1, 3, 3)
        self.register_buffer("kernel", kernel)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.l1_loss(pred, target)
        if self.grad_weight:
            loss = loss + self.grad_weight * F.l1_loss(
                F.conv2d(pred, self.kernel, padding=1, groups=3),
                F.conv2d(target, self.kernel, padding=1, groups=3),
            )
        return loss


def choose_bf16(model: nn.Module, setting: str, device: str) -> bool:
    """Use bfloat16 autocast on CPU when a quick timing test shows a clear speed-up."""
    if device != "cpu" or setting == "off":
        return False
    if setting == "on":
        return True
    probe = copy.deepcopy(model).train().to(memory_format=CHANNELS_LAST)
    x = torch.rand(4, 3, 64, 64).contiguous(memory_format=CHANNELS_LAST)

    def step(bf16: bool) -> float:
        start = time.perf_counter()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            out = probe(x)
        out.float().mean().backward()
        return time.perf_counter() - start

    step(False), step(True)
    return min(step(True) for _ in range(3)) < 0.7 * min(step(False) for _ in range(3))


def build_train_loader(cfg: type[Config], args: argparse.Namespace) -> DataLoader:
    """Synthetic on-the-fly photos, optionally mixed with real camera-noise patches."""
    length = args.iters_per_epoch * cfg.batch_size
    folders = [cfg.train_dir / "clean", *args.extra_clean]
    dataset: Dataset = OnTheFlyDataset(folders, cfg.patch_size, length)
    note = f"{len(dataset.images)} clean photos (degraded on the fly)"
    if args.real_patches.exists() and args.real_fraction > 0:
        dataset = MixedDataset(dataset, RealPatchDataset(args.real_patches, cfg.patch_size, length), args.real_fraction, length)
        note += f" + {args.real_fraction:.0%} real camera-noise patches"
    print("Training data:", note)
    return DataLoader(
        dataset, batch_size=cfg.batch_size, drop_last=True, num_workers=cfg.num_workers,
        pin_memory=cfg.device == "cuda", persistent_workers=cfg.num_workers > 0,
    )


def build_val_loaders(cfg: type[Config], args: argparse.Namespace) -> List[Tuple[str, DataLoader]]:
    """Fixed validation sets: synthetic pairs and (if prepared) held-out real pairs."""
    sets = [("synthetic", RestorationDataset(cfg.val_dir, augment=False, full_image=True))]
    if args.real_val.exists():
        sets.append(("real", RealPairDataset(args.real_val)))
    return [(name, DataLoader(ds, batch_size=1, num_workers=0)) for name, ds in sets]


@torch.inference_mode()
def validate(model: nn.Module, loader: DataLoader, device: str, bf16: bool) -> Tuple[float, float]:
    """Mean PSNR (dB) of the input and of the restored output against the clean images."""
    model.eval()
    before = after = 0.0
    for degraded, clean in loader:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            output = model(degraded.to(device).contiguous(memory_format=CHANNELS_LAST)).float().clamp_(0, 1)
        before += psnr(degraded, clean)
        after += psnr(output.cpu(), clean)
    n = len(loader)
    return before / n, after / n


def save_checkpoint(path: Path, payload: dict) -> None:
    """Write atomically so an interrupted save never corrupts the last good checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the U-Net restoration model")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--iters-per-epoch", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--warmup", type=int, default=200, help="learning-rate warm-up iterations")
    parser.add_argument("--init", type=Path, help="start from these weights (fine-tuning)")
    parser.add_argument("--grad-weight", type=float, default=0.0, help="weight of the edge-sharpness loss term")
    parser.add_argument("--real-fraction", type=float, default=0.35, help="share of real camera-noise patches")
    parser.add_argument("--real-patches", type=Path, default=Config.data_dir / "real" / "train_patches.npy")
    parser.add_argument("--real-val", type=Path, default=Config.data_dir / "real" / "val")
    parser.add_argument("--extra-clean", type=Path, nargs="*", default=[], help="more folders of clean photos")
    parser.add_argument("--bf16", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--checkpoint-dir", type=Path, default=Config.checkpoint_dir)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true", help="ignore an existing last_unet.pth")
    args = parser.parse_args(argv)

    cfg = Config
    cfg.batch_size = args.batch_size
    device = cfg.device
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    model = UNet(base=cfg.base_channels)
    last_path, best_path = args.checkpoint_dir / "last_unet.pth", args.checkpoint_dir / "best_unet.pth"
    resuming = last_path.exists() and not args.no_resume
    if args.init and not resuming:
        model.load_state_dict(load_unet_state(args.init))
        print(f"Initialised from {args.init}")
    model = model.to(device, memory_format=CHANNELS_LAST)

    bf16 = choose_bf16(model, args.bf16, device)
    use_amp = device == "cuda"
    print(f"Training on {device} | epochs={args.epochs} x {args.iters_per_epoch} iters | batch={cfg.batch_size} "
          f"| lr={args.lr} | bfloat16={bf16} | edge loss weight={args.grad_weight}")

    train_loader = build_train_loader(cfg, args)
    val_loaders = build_val_loaders(cfg, args)
    optimizer = Adam(model.parameters(), lr=args.lr)
    total_iters = args.epochs * args.iters_per_epoch
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = RestorationLoss(args.grad_weight).to(device)

    start_epoch, best = 0, float("-inf")
    if resuming:
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch, best = state["epoch"] + 1, state["best_score"]
        print(f"Resumed after epoch {start_epoch} (best score {best:.2f})")

    history = args.checkpoint_dir / "history.csv"
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not history.exists():
        history.write_text("epoch,loss,score," + ",".join(f"{n}_in,{n}_out" for n, _ in val_loaders) + ",seconds\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        started, total = time.time(), 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False, mininterval=5.0)
        for i, (degraded, clean) in enumerate(bar):
            step = epoch * args.iters_per_epoch + i
            for group in optimizer.param_groups:  # linear warm-up, then cosine decay to zero
                group["lr"] = args.lr * min(1.0, (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / total_iters))
            degraded = degraded.to(device, non_blocking=True).contiguous(memory_format=CHANNELS_LAST)
            clean = clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda" if use_amp else "cpu", dtype=torch.float16 if use_amp else torch.bfloat16, enabled=use_amp or bf16):
                prediction = model(degraded)
            loss = loss_fn(prediction.float(), clean)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        results = [validate(model, loader, device, bf16) for _, loader in val_loaders]
        score = sum(after - before for before, after in results)
        mean_loss = total / len(train_loader)
        detail = " | ".join(f"{n} {b:.2f}->{a:.2f} dB ({a - b:+.2f})" for (n, _), (b, a) in zip(val_loaders, results))
        print(f"Epoch {epoch + 1}/{args.epochs} | loss {mean_loss:.4f} | {detail} | score {score:+.2f} | {time.time() - started:.0f}s")
        with history.open("a") as f:
            f.write(f"{epoch + 1},{mean_loss:.5f},{score:.4f}," + ",".join(f"{b:.3f},{a:.3f}" for b, a in results) + f",{time.time() - started:.0f}\n")

        best = max(best, score)
        payload = {
            "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "best_score": best,
        }
        save_checkpoint(last_path, payload)
        if score >= best:
            save_checkpoint(best_path, payload)
            print(f"  new best (score {score:+.2f}) saved to {best_path}")

    print(f"Finished. Best validation score: {best:+.2f} dB of total gain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
