#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

KEY_COLS = ["source_file", "id"]
TIME_COL = "date"
DYNAMIC_HINTS = (
    "cgm", "glucose", "insulin", "bolus", "basal", "carb", "meal",
    "exercise", "activity", "steps", "heart", "hr",
)


def open_dataset(path: str) -> pads.Dataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return pads.dataset(str(p), format="parquet")


def require_columns(ds: pads.Dataset, cols: Sequence[str], label: str) -> None:
    missing = [c for c in cols if c not in ds.schema.names]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def make_group_key(source: pd.Series, ids: pd.Series) -> pd.Series:
    return source.astype(str) + "\x1f" + ids.astype(str)


def scan_dataset_identity(ds: pads.Dataset, label: str, batch_size: int) -> Dict:
    require_columns(ds, KEY_COLS + [TIME_COL], label)

    n_rows = 0
    source_counts = Counter()
    groups: Set[Tuple[str, str]] = set()
    date_min = None
    date_max = None

    scanner = ds.scanner(
        columns=KEY_COLS + [TIME_COL],
        batch_size=batch_size,
        use_threads=True,
    )

    for batch in scanner.to_batches():
        pdf = batch.to_pandas()
        n_rows += len(pdf)
        source_counts.update(pdf["source_file"].astype(str).value_counts().to_dict())
        groups.update(zip(pdf["source_file"].astype(str), pdf["id"].astype(str)))

        d = pd.to_datetime(pdf[TIME_COL], errors="coerce")
        if d.notna().any():
            bmin, bmax = d.min(), d.max()
            date_min = bmin if date_min is None or bmin < date_min else date_min
            date_max = bmax if date_max is None or bmax > date_max else date_max

    return {
        "label": label,
        "rows": int(n_rows),
        "n_groups": int(len(groups)),
        "groups": groups,
        "sources": {str(k): int(v) for k, v in sorted(source_counts.items())},
        "date_min": None if date_min is None else str(date_min),
        "date_max": None if date_max is None else str(date_max),
        "schema": {field.name: str(field.type) for field in ds.schema},
    }


def choose_sample_groups(template: pd.DataFrame, n_groups: int, seed: int) -> pd.DataFrame:
    groups = template[KEY_COLS].drop_duplicates().copy()
    rng = np.random.default_rng(seed)

    picked = []
    for _, g in groups.groupby("source_file", sort=True):
        picked.append(g.iloc[int(rng.integers(0, len(g)))])

    selected = pd.DataFrame(picked).drop_duplicates()

    if len(selected) < min(n_groups, len(groups)):
        remaining = groups.merge(selected, on=KEY_COLS, how="left", indicator=True)
        remaining = remaining[remaining["_merge"] == "left_only"][KEY_COLS]
        need = min(n_groups - len(selected), len(remaining))
        if need > 0:
            idx = rng.choice(len(remaining), size=need, replace=False)
            selected = pd.concat([selected, remaining.iloc[idx]], ignore_index=True)

    return selected.iloc[: min(n_groups, len(selected))].reset_index(drop=True)


def detect_dynamic_columns(schema_names: Sequence[str]) -> List[str]:
    dynamic = []
    for c in schema_names:
        lc = c.lower()
        if c in KEY_COLS or c == TIME_COL:
            continue
        if any(h in lc for h in DYNAMIC_HINTS):
            dynamic.append(c)
    return dynamic


def collect_sample_groups(
    ds: pads.Dataset,
    sample_groups: pd.DataFrame,
    columns: Sequence[str],
    batch_size: int,
) -> pd.DataFrame:
    wanted = set(make_group_key(sample_groups["source_file"], sample_groups["id"]).tolist())
    frames = []

    scanner = ds.scanner(columns=list(columns), batch_size=batch_size, use_threads=True)

    for batch in scanner.to_batches():
        pdf = batch.to_pandas()
        k = make_group_key(pdf["source_file"], pdf["id"])
        keep = k.isin(wanted)
        if keep.any():
            frames.append(pdf.loc[keep].copy())

    if not frames:
        raise RuntimeError("No sampled leaderboard groups were found in the test parquet.")

    out = pd.concat(frames, ignore_index=True)
    out[TIME_COL] = pd.to_datetime(out[TIME_COL])
    out["source_file"] = out["source_file"].astype(str)
    out["id"] = out["id"].astype(str)
    return out.sort_values(KEY_COLS + [TIME_COL]).reset_index(drop=True)


def numeric_nonzero_count(s: pd.Series) -> int:
    x = pd.to_numeric(s, errors="coerce")
    return int(((x.notna()) & (x != 0)).sum())


def analyze_windows(
    test_sample: pd.DataFrame,
    template: pd.DataFrame,
    sample_groups: pd.DataFrame,
    feature_cols: Sequence[str],
    origins_per_group: int,
) -> Dict:
    origin_template = template.merge(sample_groups, on=KEY_COLS, how="inner")
    origin_template[TIME_COL] = pd.to_datetime(origin_template[TIME_COL])

    per_feature = {
        c: {
            "history_nonnull_fractions": [],
            "history_nonzero_counts": [],
            "future_nonnull_fractions": [],
            "future_nonzero_counts": [],
        }
        for c in feature_cols
    }
    n_records = 0

    for (src, pid), origins in origin_template.groupby(KEY_COLS, sort=False):
        g = test_sample[
            (test_sample["source_file"] == str(src))
            & (test_sample["id"] == str(pid))
        ].sort_values(TIME_COL)

        if g.empty:
            continue

        origins = origins.sort_values(TIME_COL)
        n = min(origins_per_group, len(origins))
        if n <= 0:
            continue

        if len(origins) <= n:
            chosen = origins
        else:
            pos = np.linspace(0, len(origins) - 1, n).round().astype(int)
            chosen = origins.iloc[np.unique(pos)]

        for t in chosen[TIME_COL]:
            hist = g[(g[TIME_COL] >= t - pd.Timedelta(hours=24)) & (g[TIME_COL] <= t)]
            fut = g[(g[TIME_COL] > t) & (g[TIME_COL] <= t + pd.Timedelta(minutes=120))]

            for c in feature_cols:
                h, f = hist[c], fut[c]
                per_feature[c]["history_nonnull_fractions"].append(
                    float(h.notna().mean()) if len(h) else np.nan
                )
                per_feature[c]["history_nonzero_counts"].append(numeric_nonzero_count(h))
                per_feature[c]["future_nonnull_fractions"].append(
                    float(f.notna().mean()) if len(f) else np.nan
                )
                per_feature[c]["future_nonzero_counts"].append(numeric_nonzero_count(f))

            n_records += 1

    def safe_mean(xs):
        arr = np.asarray(xs, dtype=float)
        return None if len(arr) == 0 or np.all(np.isnan(arr)) else float(np.nanmean(arr))

    def safe_median(xs):
        arr = np.asarray(xs, dtype=float)
        return None if len(arr) == 0 or np.all(np.isnan(arr)) else float(np.nanmedian(arr))

    summary = {}
    for c, vals in per_feature.items():
        summary[c] = {
            "history_mean_nonnull_fraction": safe_mean(vals["history_nonnull_fractions"]),
            "history_median_nonzero_count_per_24h": safe_median(vals["history_nonzero_counts"]),
            "future_mean_nonnull_fraction": safe_mean(vals["future_nonnull_fractions"]),
            "future_median_nonzero_count_per_120m": safe_median(vals["future_nonzero_counts"]),
        }

    return {
        "n_sampled_origins": int(n_records),
        "expected_history_rows_on_5min_grid_including_t": 289,
        "expected_future_rows_on_5min_grid": 24,
        "features": summary,
    }


def make_persistence_submission(
    test_ds: pads.Dataset,
    template_path: str,
    out_path: str,
    batch_size: int,
    cgm_col: str = "CGM",
) -> Dict:
    if cgm_col not in test_ds.schema.names:
        raise ValueError(f"Cannot build persistence submission: '{cgm_col}' not found.")

    pred_cols = ["pred_30", "pred_60", "pred_90", "pred_120"]
    template = pd.read_parquet(template_path, columns=KEY_COLS + [TIME_COL] + pred_cols)
    template[TIME_COL] = pd.to_datetime(template[TIME_COL])
    template["source_file"] = template["source_file"].astype(str)
    template["id"] = template["id"].astype(str)

    lookup = pd.MultiIndex.from_frame(template[KEY_COLS + [TIME_COL]])
    if lookup.has_duplicates:
        raise ValueError("Template key (source_file, id, date) is not unique.")

    current_cgm = np.full(len(template), np.nan, dtype=np.float64)

    scanner = test_ds.scanner(
        columns=KEY_COLS + [TIME_COL, cgm_col],
        batch_size=batch_size,
        use_threads=True,
    )

    matched = 0
    for batch in scanner.to_batches():
        pdf = batch.to_pandas()
        pdf[TIME_COL] = pd.to_datetime(pdf[TIME_COL])
        pdf["source_file"] = pdf["source_file"].astype(str)
        pdf["id"] = pdf["id"].astype(str)

        idx = pd.MultiIndex.from_frame(pdf[KEY_COLS + [TIME_COL]])
        loc = lookup.get_indexer(idx)
        keep = loc >= 0
        if keep.any():
            vals = pd.to_numeric(pdf.loc[keep, cgm_col], errors="coerce").to_numpy(float)
            current_cgm[loc[keep]] = vals
            matched += int(keep.sum())

    missing = int(np.isnan(current_cgm).sum())
    if missing:
        raise RuntimeError(
            f"Matched {matched:,} origins but {missing:,} template rows still lack current CGM."
        )

    for c in pred_cols:
        template[c] = current_cgm

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    template.to_parquet(out, index=False)

    return {
        "path": str(out),
        "rows": int(len(template)),
        "matched_origins": int(matched),
        "missing_current_cgm": missing,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out-json", default="live_probe.json")
    ap.add_argument("--sample-groups", type=int, default=24)
    ap.add_argument("--origins-per-group", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--persistence-out", default=None)
    args = ap.parse_args()

    train_ds = open_dataset(args.train)
    test_ds = open_dataset(args.test)

    print("\n=== 1. Train ===")
    train_info = scan_dataset_identity(train_ds, "train", args.batch_size)
    train_groups = train_info.pop("groups")
    print(
        f"{train_info['rows']:,} rows | {train_info['n_groups']:,} groups | "
        f"{len(train_info['sources'])} sources"
    )

    print("\n=== 2. Test ===")
    test_info = scan_dataset_identity(test_ds, "test", args.batch_size)
    _test_groups = test_info.pop("groups")
    print(
        f"{test_info['rows']:,} rows | {test_info['n_groups']:,} groups | "
        f"{len(test_info['sources'])} sources"
    )

    print("\n=== 3. Live leaderboard ===")
    template = pd.read_parquet(args.template, columns=KEY_COLS + [TIME_COL])
    template[TIME_COL] = pd.to_datetime(template[TIME_COL])
    template["source_file"] = template["source_file"].astype(str)
    template["id"] = template["id"].astype(str)

    live_groups = template[KEY_COLS].drop_duplicates()
    live_group_tuples = set(zip(live_groups["source_file"], live_groups["id"]))
    known_groups = live_group_tuples & train_groups
    novel_groups = live_group_tuples - train_groups

    known_df = pd.DataFrame(list(known_groups), columns=KEY_COLS)
    known_df["known"] = True
    if len(known_df):
        t2 = template.merge(known_df, on=KEY_COLS, how="left")
        known_row_mask = t2["known"].fillna(False).astype(bool)
    else:
        known_row_mask = pd.Series(False, index=template.index)

    source_rows = template["source_file"].value_counts().sort_index()
    source_groups = live_groups["source_file"].value_counts().sort_index()

    print(
        f"{len(template):,} origins | {len(live_groups):,} groups | "
        f"{template['source_file'].nunique()} sources"
    )
    print(
        f"known groups: {len(known_groups):,}/{len(live_groups):,} "
        f"({100*len(known_groups)/max(len(live_groups),1):.1f}%)"
    )
    print(
        f"known rows:   {int(known_row_mask.sum()):,}/{len(template):,} "
        f"({100*known_row_mask.mean():.1f}%)"
    )

    print("\nLive source composition:")
    for src in source_rows.index:
        print(
            f"  {src:18s} rows={int(source_rows[src]):9,d} "
            f"({100*source_rows[src]/len(template):5.1f}%)  "
            f"groups={int(source_groups.get(src, 0)):3d}"
        )

    print("\n=== 4. Dynamic-looking columns shared by train/test ===")
    dynamic_cols = detect_dynamic_columns(test_ds.schema.names)
    common_dynamic = [c for c in dynamic_cols if c in train_ds.schema.names]
    for c in common_dynamic:
        print(f"  {c:30s} {test_ds.schema.field(c).type}")

    print("\n=== 5. Real leaderboard-window probe ===")
    sample_groups = choose_sample_groups(template, args.sample_groups, args.seed)
    sample_cols = KEY_COLS + [TIME_COL] + common_dynamic
    test_sample = collect_sample_groups(test_ds, sample_groups, sample_cols, args.batch_size)
    window_info = analyze_windows(
        test_sample, template, sample_groups, common_dynamic, args.origins_per_group
    )

    print(f"Analyzed {window_info['n_sampled_origins']:,} origins.")
    for c, s in window_info["features"].items():
        h = s["history_mean_nonnull_fraction"]
        f = s["future_mean_nonnull_fraction"]
        hnz = s["history_median_nonzero_count_per_24h"]
        fnz = s["future_median_nonzero_count_per_120m"]
        htxt = "n/a" if h is None else f"{100*h:5.1f}%"
        ftxt = "n/a" if f is None else f"{100*f:5.1f}%"
        hnztxt = "n/a" if hnz is None else f"{hnz:.1f}"
        fnztxt = "n/a" if fnz is None else f"{fnz:.1f}"
        print(
            f"  {c:28s} hist non-null={htxt} hist nonzero~{hnztxt:>6s} | "
            f"future non-null={ftxt} future nonzero~{fnztxt:>6s}"
        )

    report = {
        "train": train_info,
        "test": test_info,
        "live": {
            "rows": int(len(template)),
            "groups": int(len(live_groups)),
            "sources": int(template["source_file"].nunique()),
            "known_groups": int(len(known_groups)),
            "novel_groups": int(len(novel_groups)),
            "known_group_fraction": float(len(known_groups) / max(len(live_groups), 1)),
            "known_rows": int(known_row_mask.sum()),
            "novel_rows": int((~known_row_mask).sum()),
            "known_row_fraction": float(known_row_mask.mean()),
            "source_rows": {str(k): int(v) for k, v in source_rows.items()},
            "source_groups": {str(k): int(v) for k, v in source_groups.items()},
        },
        "dynamic_columns_shared_train_test": common_dynamic,
        "window_probe": window_info,
    }

    if args.persistence_out:
        print("\n=== 6. Persistence submission ===")
        pinfo = make_persistence_submission(
            test_ds, args.template, args.persistence_out, args.batch_size
        )
        report["persistence_submission"] = pinfo
        print(f"Wrote {pinfo['path']} ({pinfo['rows']:,} rows)")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nWrote report: {out_json}")
    print("\nImportant:")
    print("  Future CGM in the public test parquet is target information; never use it as input.")
    print("  The future-coverage section tells us which non-CGM covariates are actually")
    print("  available for a conditional forecaster.")
    if args.persistence_out:
        print("\nScore from the submission-toolkit repo with:")
        print(f'  python run.py "{args.persistence_out}" --competition live --horizon all')


if __name__ == "__main__":
    main()
