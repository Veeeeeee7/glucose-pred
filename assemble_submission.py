#!/usr/bin/env python3
"""
assemble_submission.py

Merge per-shard prediction parquets into ONE template-complete submission.

`predict_jepa.py --only-source <ds>` writes a partial parquet per shard so that
concurrent cluster jobs never contend for one file. `run.py` accepts only a file
carrying exactly the template's rows in exactly the template's order, so the
shards have to be reassembled. That is this script, and the script -- not the
merged file -- is the version-controlled artifact: the output is derived and
disposable, and re-running is always cheaper than trusting a stale copy.

It never edits or moves a shard. It reads them, reindexes onto the template, and
writes a new file alongside them plus a PROVENANCE.md naming every input.

The merge is by (id, source_file, date), not by position: a shard's rows are a
subset in its own order, and aligning them positionally would silently transpose
predictions between participants. Overlapping shards are an error rather than a
last-writer-wins merge, because two shards covering one row means the fan-out was
misconfigured and the quieter of the two answers is not obviously the right one.

Usage:
  python assemble_submission.py --date YYYY-MM-DD
  python assemble_submission.py --date YYYY-MM-DD --competition annual
  python assemble_submission.py --date YYYY-MM-DD --tag jepa_x_cgm_jepa_ridge_interp
  python assemble_submission.py --date YYYY-MM-DD --check     # report, write nothing
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["id", "source_file", "date"]
PRED_COLS = ["pred_30", "pred_60", "pred_90", "pred_120"]
EXPERIMENT = "jepa_zeroshot"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="Results partition to assemble, YYYY-MM-DD")
    ap.add_argument("--competition", default="live", choices=["live", "annual"])
    ap.add_argument("--tag", default=None,
                    help="Assemble only shards whose filename starts with this tag "
                         "(e.g. jepa_x_cgm_jepa_ridge_interp). Required when a tree holds "
                         "more than one configuration.")
    ap.add_argument("--data-dir", default=None, help="overrides $DATA_DIR / data")
    ap.add_argument("--check", action="store_true", help="report coverage, write nothing")
    args = ap.parse_args()

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        sys.exit(f"--date must be YYYY-MM-DD, got {args.date!r}")

    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir or os.environ.get("DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    comp_dir = data_dir / ("live_leaderboard" if args.competition == "live" else "annual_competition")
    template_path = comp_dir / "template.parquet"
    pred_dir = root / "results" / args.date / EXPERIMENT / "preds"

    for p in (template_path, pred_dir):
        if not p.exists():
            sys.exit(f"Not found: {p}")

    shards = sorted(p for p in pred_dir.glob(f"*{args.competition}full*.parquet")
                    if "_assembled" not in p.name)
    if args.tag:
        shards = [p for p in shards if p.name.startswith(args.tag)]
    if not shards:
        sys.exit(f"No --scope full shards in {pred_dir}"
                 + (f" matching tag {args.tag!r}" if args.tag else ""))

    # One tree can hold several configurations; merging across them would silently
    # interleave two models' predictions into one submission.
    configs = defaultdict(list)
    for p in shards:
        configs[p.name.split(f"_{args.competition}full")[0]].append(p)
    if len(configs) > 1 and not args.tag:
        sys.exit("This tree holds shards from more than one configuration:\n  "
                 + "\n  ".join(sorted(configs))
                 + "\nRe-run with --tag <one of them>.")
    tag = args.tag or next(iter(configs))

    template = pd.read_parquet(template_path)
    print(f"template : {len(template):,} rows  ({template_path.relative_to(root)})")
    print(f"shards   : {len(shards)} for {tag}")

    frames = []
    for p in shards:
        d = pd.read_parquet(p)
        missing = [c for c in KEY + PRED_COLS if c not in d.columns]
        if missing:
            sys.exit(f"{p.name} is missing columns: {missing}")
        print(f"  {p.name}: {len(d):,} rows")
        frames.append(d[KEY + PRED_COLS])

    merged = pd.concat(frames, ignore_index=True)

    dup = merged.duplicated(subset=KEY, keep=False)
    if dup.any():
        sample = merged.loc[dup, KEY].head(5).to_string(index=False)
        sys.exit(f"ERROR: {int(dup.sum()):,} rows are covered by more than one shard.\n"
                 f"The shards overlap, so the fan-out was misconfigured. First few:\n{sample}")

    # Reindex onto the template: this is what enforces the row set AND the order.
    out = template[KEY].merge(merged, on=KEY, how="left", validate="one_to_one")
    assert len(out) == len(template), "merge changed the row count"

    covered = out[PRED_COLS].notna().any(axis=1)
    print(f"\ncovered  : {int(covered.sum()):,} / {len(out):,} template rows "
          f"({100 * covered.mean():.2f}%)")

    if not covered.all():
        gaps = out.loc[~covered, "source_file"].value_counts()
        print("uncovered rows by source_file:")
        print("  " + "\n  ".join(f"{k}: {v:,}" for k, v in gaps.items()))
        print("\nERROR: the submission is incomplete. run.py rejects any populated "
              "column containing NaN, so fill these before submitting: re-run the "
              "missing shards, or run without --only-source.")
        if not args.check:
            sys.exit(1)

    if args.check:
        print("\n--check: nothing written.")
        return

    out_path = pred_dir / f"{tag}_{args.competition}full_assembled.parquet"
    out.to_parquet(out_path, index=False)

    prov = pred_dir / f"PROVENANCE_{tag}_{args.competition}full.md"
    prov.write_text(
        f"# {out_path.name}\n\n"
        f"Generated by `assemble_submission.py` on "
        f"{datetime.now().isoformat(timespec='seconds')}.\n"
        f"Derived and disposable -- the script is the artifact, not this file.\n\n"
        f"- results date: `{args.date}`\n"
        f"- competition: `{args.competition}`\n"
        f"- configuration: `{tag}`\n"
        f"- template: `{template_path.relative_to(root)}` ({len(template):,} rows)\n"
        f"- coverage: {int(covered.sum()):,} / {len(out):,} rows\n\n"
        f"## Shards merged\n\n"
        + "".join(f"- `{p.name}` ({len(pd.read_parquet(p)):,} rows)\n" for p in shards),
        encoding="utf-8",
    )

    print(f"\nwrote    : {out_path.relative_to(root)}")
    print(f"provenance: {prov.relative_to(root)}")
    print(f"\nvalidate with:\n  python run.py {out_path.relative_to(root)} "
          f"--competition {args.competition} --horizon all")


if __name__ == "__main__":
    main()
