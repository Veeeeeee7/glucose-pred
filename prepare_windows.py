#!/usr/bin/env python3
"""
prepare_windows_fixed.py

Standalone MetaboNet training-window builder.

- Streams train.parquet once.
- Handles participants that appear in multiple non-contiguous parquet blocks.
- Never joins separated blocks and invents continuity.
- Keeps every occurrence of the same (source_file,id) in the same train/val split.
- Caps windows per participant across all of that participant's segments.
- Builds:
    history: 288 x [CGM, insulin, basal, bolus, carbs]
    future covariates: 24 x [insulin, basal, bolus, carbs]
    target: 24 x CGM
- Candidate origins every 15 minutes.
- Optional live-like eligibility:
    CGM gaps <= 30 min
    insulin gaps <= 120 min
    >=1 carb event in prior 24 h
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

KEY = ["source_file", "id"]
TIME = "date"

HIST_FEATURES = ["CGM", "insulin", "basal", "bolus", "carbs"]
FUTURE_FEATURES = ["insulin", "basal", "bolus", "carbs"]

HIST_LEN = 288
FUT_LEN = 24

# 0-based positions inside the 24-step future:
# +30,+60,+90,+120 minutes = steps 6,12,18,24 => indices 5,11,17,23
SCORED_FUTURE_INDEX = np.array([5, 11, 17, 23], dtype=np.int64)


def stable_u01(source: str, pid: str, seed: int) -> float:
    payload = f"{seed}|{source}|{pid}".encode("utf-8")
    x = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return x / float(2**64)


def max_missing_run(mask: np.ndarray) -> int:
    """Maximum consecutive False values in a 1-D boolean mask."""
    if mask.size == 0:
        return 0
    bad = ~mask
    if not bad.any():
        return 0

    x = np.concatenate(([False], bad, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(x))
    lengths = edges[1::2] - edges[::2]
    return int(lengths.max()) if len(lengths) else 0


class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)

        self.buffers = {
            "hist": [],
            "fut_cov": [],
            "target": [],
            "origin_minute": [],
            "source_code": [],
        }

        self.shard_idx = 0
        self.n_written = 0

    def add(self, hist, fut_cov, target, origin_minute, source_code):
        self.buffers["hist"].append(hist.astype(np.float32, copy=False))
        self.buffers["fut_cov"].append(fut_cov.astype(np.float32, copy=False))
        self.buffers["target"].append(target.astype(np.float32, copy=False))
        self.buffers["origin_minute"].append(np.int16(origin_minute))
        self.buffers["source_code"].append(np.int16(source_code))

        if len(self.buffers["hist"]) >= self.shard_size:
            self.flush()

    def flush(self):
        n = len(self.buffers["hist"])
        if n == 0:
            return

        out = self.out_dir / f"shard_{self.shard_idx:05d}.npz"

        np.savez(
            out,
            hist=np.stack(self.buffers["hist"]),
            fut_cov=np.stack(self.buffers["fut_cov"]),
            target=np.stack(self.buffers["target"]),
            origin_minute=np.asarray(
                self.buffers["origin_minute"], dtype=np.int16
            ),
            source_code=np.asarray(
                self.buffers["source_code"], dtype=np.int16
            ),
        )

        self.n_written += n
        self.shard_idx += 1

        for v in self.buffers.values():
            v.clear()

    def close(self):
        self.flush()


def exact_five_minute_grid(
    times_ns: np.ndarray,
    start: int,
    end: int,
) -> bool:
    """
    Require every adjacent timestamp in inclusive slice [start,end]
    to be exactly 5 minutes apart.

    This is stricter and safer than checking only total elapsed duration.
    """
    seg = times_ns[start : end + 1]
    if len(seg) <= 1:
        return True

    expected = 5 * 60 * 1_000_000_000
    return bool(np.all(np.diff(seg) == expected))


def count_eligible_origins(
    g: pd.DataFrame,
    live_like: bool,
) -> Tuple[np.ndarray, Counter]:
    stats = Counter()

    if len(g) < HIST_LEN + FUT_LEN:
        stats["too_short_segment"] += 1
        return np.empty(0, dtype=np.int64), stats

    g = (
        g.sort_values(TIME)
        .drop_duplicates(subset=[TIME], keep="last")
        .reset_index(drop=True)
    )

    t = pd.to_datetime(g[TIME], errors="coerce")
    valid_t = t.notna().to_numpy()

    if not valid_t.all():
        g = g.loc[valid_t].reset_index(drop=True)
        t = pd.to_datetime(g[TIME])

    if len(g) < HIST_LEN + FUT_LEN:
        stats["too_short_segment"] += 1
        return np.empty(0, dtype=np.int64), stats

    times_ns = t.astype("int64").to_numpy()

    vals = (
        g[HIST_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float32)
    )

    candidates = np.arange(
        HIST_LEN - 1,
        len(g) - FUT_LEN,
        dtype=np.int64,
    )

    # Match leaderboard's 15-minute origin cadence.
    minute = t.dt.minute.to_numpy()
    second = t.dt.second.to_numpy()
    candidates = candidates[
        (minute[candidates] % 15 == 0)
        & (second[candidates] == 0)
    ]

    stats["candidate_origins"] += int(len(candidates))

    eligible = []

    for i in candidates:
        h0 = i - HIST_LEN + 1
        f1 = i + FUT_LEN

        if not exact_five_minute_grid(times_ns, h0, f1):
            stats["reject_noncontiguous_grid"] += 1
            continue

        hist = vals[h0 : i + 1]
        fut = vals[i + 1 : f1 + 1]
        target = fut[:, 0]

        # All four scored target times must exist.
        if not np.isfinite(target[SCORED_FUTURE_INDEX]).all():
            stats["reject_missing_scored_target"] += 1
            continue

        # Current CGM must exist.
        if not np.isfinite(hist[-1, 0]):
            stats["reject_missing_current_cgm"] += 1
            continue

        if live_like:
            cgm_mask = np.isfinite(hist[:, 0])
            insulin_mask = np.isfinite(hist[:, 1])

            # <=30 min between CGM observations:
            # at most 5 missing 5-min bins between observed values.
            if max_missing_run(cgm_mask) > 5:
                stats["reject_cgm_gap"] += 1
                continue

            # <=120 min between insulin observations:
            # at most 23 missing 5-min bins.
            if max_missing_run(insulin_mask) > 23:
                stats["reject_insulin_gap"] += 1
                continue

            carbs = hist[:, 4]
            if not np.any(np.isfinite(carbs) & (carbs > 0)):
                stats["reject_no_carb_event"] += 1
                continue

        eligible.append(i)

    stats["eligible_origins"] += int(len(eligible))

    return np.asarray(eligible, dtype=np.int64), stats


def write_selected_windows(
    g: pd.DataFrame,
    eligible: np.ndarray,
    chosen: np.ndarray,
    source_code: int,
    writer: ShardWriter,
) -> int:
    if len(chosen) == 0:
        return 0

    g = (
        g.sort_values(TIME)
        .drop_duplicates(subset=[TIME], keep="last")
        .reset_index(drop=True)
    )

    t = pd.to_datetime(g[TIME], errors="coerce")
    valid_t = t.notna().to_numpy()

    if not valid_t.all():
        g = g.loc[valid_t].reset_index(drop=True)
        t = pd.to_datetime(g[TIME])

    vals = (
        g[HIST_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float32)
    )

    for i in chosen:
        h0 = i - HIST_LEN + 1
        f1 = i + FUT_LEN

        hist = vals[h0 : i + 1]      # [288,5]
        fut = vals[i + 1 : f1 + 1]   # [24,5]

        fut_cov = fut[:, 1:]          # [24,4]
        target = fut[:, 0]            # [24]

        origin = t.iloc[i]
        origin_minute = int(
            origin.hour * 60 + origin.minute
        )

        writer.add(
            hist,
            fut_cov,
            target,
            origin_minute,
            source_code,
        )

    return int(len(chosen))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--train",
        required=True,
        help="MetaboNet train.parquet",
    )
    ap.add_argument(
        "--out",
        default="model_data_v1",
    )
    ap.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
    )
    ap.add_argument(
        "--max-windows-per-group",
        type=int,
        default=100,
        help=(
            "Maximum total windows per (source_file,id), "
            "across all of its physical parquet segments. "
            "0 means unlimited."
        ),
    )
    ap.add_argument(
        "--shard-size",
        type=int,
        default=2048,
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=500_000,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    ap.add_argument(
        "--no-live-like",
        action="store_true",
        help=(
            "Disable leaderboard-like history filters "
            "(CGM gap, insulin gap, carb event)."
        ),
    )

    args = ap.parse_args()

    path = Path(args.train)
    out = Path(args.out)

    if not path.exists():
        raise FileNotFoundError(path)

    if not (0 < args.val_fraction < 1):
        raise ValueError("--val-fraction must be in (0,1)")

    # Avoid mixing old partial shards from a crashed run.
    if (out / "train").exists() or (out / "val").exists():
        raise RuntimeError(
            f"{out} already contains train/val output. "
            "Delete that directory before rerunning."
        )

    pf = pq.ParquetFile(path)

    needed = KEY + [TIME] + HIST_FEATURES
    missing = [c for c in needed if c not in pf.schema.names]

    if missing:
        raise ValueError(
            f"train parquet is missing required columns: {missing}"
        )

    train_writer = ShardWriter(
        out / "train",
        args.shard_size,
    )
    val_writer = ShardWriter(
        out / "val",
        args.shard_size,
    )

    source_to_code: Dict[str, int] = {}
    source_split_counts = Counter()
    rejection_counts = Counter()

    # Number of physical contiguous segments seen for each participant.
    segment_counts: Counter[Tuple[str, str]] = Counter()

    # Total number of windows already written for each participant,
    # enforcing the cap across all physical segments.
    group_written_counts: Counter[Tuple[str, str]] = Counter()

    rng = np.random.default_rng(args.seed)

    carry = pd.DataFrame()
    n_segments = 0

    def process_segment(
        seg: pd.DataFrame,
        key: Tuple[str, str],
    ) -> None:
        nonlocal n_segments

        source, pid = key
        segment_counts[key] += 1
        n_segments += 1

        if source not in source_to_code:
            source_to_code[source] = len(source_to_code)

        is_val = (
            stable_u01(source, pid, args.seed)
            < args.val_fraction
        )
        split = "val" if is_val else "train"
        writer = val_writer if is_val else train_writer

        if args.max_windows_per_group > 0:
            remaining = (
                args.max_windows_per_group
                - group_written_counts[key]
            )
            if remaining <= 0:
                return
        else:
            remaining = None

        eligible, stats = count_eligible_origins(
            seg,
            live_like=not args.no_live_like,
        )

        rejection_counts.update(stats)

        if len(eligible) == 0:
            return

        if (
            remaining is not None
            and len(eligible) > remaining
        ):
            chosen = np.sort(
                rng.choice(
                    eligible,
                    size=remaining,
                    replace=False,
                )
            )
        else:
            chosen = eligible

        written = write_selected_windows(
            seg,
            eligible,
            chosen,
            source_to_code[source],
            writer,
        )

        group_written_counts[key] += written
        source_split_counts[(source, split)] += written

    print(
        "Streaming train.parquet once. "
        "Non-contiguous participant blocks are handled safely."
    )

    for batch_idx, batch in enumerate(
        pf.iter_batches(
            batch_size=args.batch_size,
            columns=needed,
            use_threads=True,
        )
    ):
        df = batch.to_pandas()

        if not carry.empty:
            df = pd.concat(
                [carry, df],
                ignore_index=True,
            )
            carry = pd.DataFrame()

        if df.empty:
            continue

        df["source_file"] = (
            df["source_file"].astype(str)
        )
        df["id"] = df["id"].astype(str)

        keys = list(
            zip(
                df["source_file"].to_numpy(),
                df["id"].to_numpy(),
            )
        )

        boundaries = [0]

        for j in range(1, len(keys)):
            if keys[j] != keys[j - 1]:
                boundaries.append(j)

        boundaries.append(len(df))

        # Process all complete contiguous segments except the final one.
        for b in range(len(boundaries) - 2):
            a = boundaries[b]
            z = boundaries[b + 1]

            process_segment(
                df.iloc[a:z],
                keys[a],
            )

        # Hold final segment because it may continue in next parquet batch.
        carry = df.iloc[
            boundaries[-2] :
        ].copy()

        if (batch_idx + 1) % 20 == 0:
            train_n = (
                train_writer.n_written
                + len(train_writer.buffers["hist"])
            )
            val_n = (
                val_writer.n_written
                + len(val_writer.buffers["hist"])
            )
            repeated = sum(
                v > 1
                for v in segment_counts.values()
            )

            print(
                f"  batches={batch_idx+1:,} "
                f"segments={n_segments:,} "
                f"unique_groups={len(segment_counts):,} "
                f"repeated_groups={repeated:,} "
                f"train_windows={train_n:,} "
                f"val_windows={val_n:,}"
            )

    if not carry.empty:
        key = (
            str(carry["source_file"].iloc[0]),
            str(carry["id"].iloc[0]),
        )
        process_segment(
            carry,
            key,
        )

    train_writer.close()
    val_writer.close()

    repeated_groups = {
        k: v
        for k, v in segment_counts.items()
        if v > 1
    }

    repeated_examples = [
        {
            "source_file": k[0],
            "id": k[1],
            "segments": int(v),
        }
        for k, v in list(
            sorted(repeated_groups.items())
        )[:25]
    ]

    metadata = {
        "history_features": HIST_FEATURES,
        "future_features": FUTURE_FEATURES,
        "history_length": HIST_LEN,
        "future_length": FUT_LEN,
        "scored_future_indices_zero_based":
            SCORED_FUTURE_INDEX.tolist(),
        "scored_horizons_minutes":
            [30, 60, 90, 120],
        "train_windows":
            train_writer.n_written,
        "val_windows":
            val_writer.n_written,
        "unique_groups_seen":
            len(segment_counts),
        "contiguous_segments_seen":
            n_segments,
        "groups_appearing_in_multiple_segments":
            len(repeated_groups),
        "repeated_group_examples":
            repeated_examples,
        "val_fraction_by_group_hash":
            args.val_fraction,
        "max_windows_per_group_across_all_segments":
            args.max_windows_per_group,
        "live_like_filter":
            not args.no_live_like,
        "source_to_code":
            source_to_code,
        "source_window_counts": {
            f"{source}|{split}": int(n)
            for (source, split), n
            in sorted(source_split_counts.items())
        },
        "window_filter_counts": {
            k: int(v)
            for k, v
            in rejection_counts.items()
        },
    }

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    (out / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nDone.")
    print(
        f"  train windows: "
        f"{train_writer.n_written:,}"
    )
    print(
        f"  val windows:   "
        f"{val_writer.n_written:,}"
    )
    print(
        f"  unique groups: "
        f"{len(segment_counts):,}"
    )
    print(
        f"  segments:      "
        f"{n_segments:,}"
    )
    print(
        "  repeated groups handled conservatively: "
        f"{len(repeated_groups):,}"
    )
    print(
        f"  metadata:      "
        f"{out / 'metadata.json'}"
    )


if __name__ == "__main__":
    main()
