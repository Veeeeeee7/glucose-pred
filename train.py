#!/usr/bin/env python3
"""
train.py

Train the causal Gaussian dual-stream forecaster on NPZ shards created by
prepare_windows.py.

Validation is group-disjoint because prepare_windows.py hashes whole
(source_file,id) groups into train or val. This prevents heavily overlapping
24-hour windows from the same participant appearing on both sides.

The loss supervises all available 5-min future CGM points, with additional
weight on the four scored competition horizons (+30,+60,+90,+120).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

from model import DualStreamForecaster

SCORED = [5, 11, 17, 23]
HORIZONS = [30, 60, 90, 120]


class ShardIterable(IterableDataset):
    def __init__(self, shard_dir: Path, shuffle: bool, seed: int):
        super().__init__()
        self.shards = sorted(Path(shard_dir).glob("shard_*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"No shard_*.npz files in {shard_dir}")
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        files = list(self.shards)

        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(files)

        if worker is not None:
            files = files[worker.id::worker.num_workers]

        for path in files:
            with np.load(path, allow_pickle=False) as d:
                hist = d["hist"]
                fut_cov = d["fut_cov"]
                target = d["target"]
                origin_minute = d["origin_minute"]
                source_code = d["source_code"]

                idx = np.arange(len(hist))
                if self.shuffle:
                    rng.shuffle(idx)

                for i in idx:
                    yield (
                        torch.from_numpy(hist[i]),
                        torch.from_numpy(fut_cov[i]),
                        torch.from_numpy(target[i]),
                        torch.tensor(origin_minute[i], dtype=torch.float32),
                        torch.tensor(source_code[i], dtype=torch.int64),
                    )


def weighted_masked_mse(pred, target):
    mask = torch.isfinite(target)
    y = torch.nan_to_num(target, nan=0.0)
    weights = torch.ones(24, device=pred.device, dtype=pred.dtype)
    weights[SCORED] = 2.0
    w = weights.unsqueeze(0) * mask.float()
    return (((pred - y) ** 2) * w).sum() / w.sum().clamp_min(1.0)


@torch.no_grad()
def validate(model, loader, device, code_to_source):
    model.eval()

    sq = {h: 0.0 for h in HORIZONS}
    n = {h: 0 for h in HORIZONS}

    source_sq = {}
    source_n = {}

    for hist, fut_cov, target, origin_minute, source_code in loader:
        hist = hist.to(device, non_blocking=True)
        fut_cov = fut_cov.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        origin_minute = origin_minute.to(device, non_blocking=True)

        pred = model(hist, fut_cov, origin_minute)

        for h, j in zip(HORIZONS, SCORED):
            valid = torch.isfinite(target[:, j])
            if valid.any():
                err2 = (pred[valid, j] - target[valid, j]) ** 2
                sq[h] += float(err2.sum().item())
                n[h] += int(valid.sum().item())

        # Per-source pooled scored-horizon MSE.
        for row in range(len(source_code)):
            code = int(source_code[row].item())
            src = code_to_source.get(code, str(code))
            source_sq.setdefault(src, 0.0)
            source_n.setdefault(src, 0)
            for j in SCORED:
                if torch.isfinite(target[row, j]):
                    e = float((pred[row, j] - target[row, j]).item())
                    source_sq[src] += e * e
                    source_n[src] += 1

    rmse = {
        h: math.sqrt(sq[h] / n[h]) if n[h] else float("nan")
        for h in HORIZONS
    }
    overall_sq = sum(sq.values())
    overall_n = sum(n.values())
    overall_rmse = math.sqrt(overall_sq / overall_n) if overall_n else float("nan")

    source_rmse = {
        src: math.sqrt(source_sq[src] / source_n[src])
        for src in source_sq
        if source_n[src] > 0
    }
    return rmse, overall_rmse, source_rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="model_data_v1")
    ap.add_argument("--out", default="checkpoints/gluco_v1.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data_root = Path(args.data)
    metadata = json.loads((data_root / "metadata.json").read_text(encoding="utf-8"))
    code_to_source = {int(v): k for k, v in metadata["source_to_code"].items()}

    train_ds = ShardIterable(data_root / "train", shuffle=True, seed=args.seed)
    val_ds = ShardIterable(data_root / "val", shuffle=False, seed=args.seed + 1000)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = DualStreamForecaster().to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    print(f"device={device}")
    print(f"train windows={metadata['train_windows']:,}")
    print(f"val windows={metadata['val_windows']:,}")
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        running = 0.0
        steps = 0

        for hist, fut_cov, target, origin_minute, _source_code in train_loader:
            hist = hist.to(device, non_blocking=True)
            fut_cov = fut_cov.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            origin_minute = origin_minute.to(device, non_blocking=True)

            pred = model(hist, fut_cov, origin_minute)
            loss = weighted_masked_mse(pred, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += float(loss.item())
            steps += 1

        rmse, overall, source_rmse = validate(
            model, val_loader, device, code_to_source
        )

        sigma = float(model.decomposer.sigma.detach().cpu().item())
        print(
            f"\nepoch {epoch:02d} | train MSE={running/max(steps,1):.3f} "
            f"| val overall RMSE={overall:.2f} | sigma={sigma:.2f} steps"
        )
        print(
            "  " + "  ".join(
                f"{h}m={rmse[h]:.2f}" for h in HORIZONS
            )
        )

        top_sources = sorted(source_rmse.items(), key=lambda kv: kv[0])
        if top_sources:
            print("  per-source scored RMSE:")
            print("    " + " | ".join(f"{s}={r:.2f}" for s, r in top_sources))

        if overall < best:
            best = overall
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_overall_rmse": overall,
                    "val_horizon_rmse": rmse,
                    "metadata": metadata,
                },
                out,
            )
            print(f"  saved best -> {out}")

    print(f"\nbest val overall RMSE={best:.2f}")


if __name__ == "__main__":
    main()
