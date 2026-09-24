#!/usr/bin/env python3
"""
predict_jepa.py

Zero-shot-encoder glucose forecasting baseline for the MetaboNet benchmark.

The CGM-JEPA encoder stays frozen: no gradient step is taken and no pretrained
weight is modified. Because the published checkpoints contain an encoder only --
the pretraining loss lives entirely in 96-d latent space and no decoder back to
mg/dL was ever released -- a forecast needs a readout on top of the frozen
embeddings. This driver fits the smallest one that is defensible: a closed-form
ridge regression, the same frozen-encoder-plus-linear-probe protocol the paper
uses for classification, moved to regression.

The head predicts the *change* from the current glucose value, not the level:

    G(t+h) = G(t) + w_h . phi(x)

so persistence is the zero solution. A head that learns nothing degrades to
persistence instead of to garbage, and the gap between the two rows in the
results file is exactly what the JEPA representation contributed.

Features phi(x) are the frozen encoder's last-patch embedding (the most recent
hour) concatenated with its mean-pooled day embedding: 192 dims, standardized.
The ridge penalty is picked on a participant-disjoint split inside the fit set.

Modes:
  --scope sub     score a stratified subsample against live targets (not submittable)
  --scope full    write a template-complete parquet for run.py / the leaderboard
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from jepa_model import check_device_parity, embed_windows, load_encoder, resolve_device
from jepa_windows import (
    EVAL_CHUNK, HORIZONS, TEMPLATE_KEY, build_fit_set, iter_eval_chunks, read_template,
    sample_template_rows,
)
from metrics import calculate_dts_error_grid, calculate_mard, calculate_rmse

EXPERIMENT = "jepa_zeroshot"
PRED_COLS = [f"pred_{h}" for h in HORIZONS]
TARGET_COLS = [f"target_{h}" for h in HORIZONS]
GLUCOSE_RANGE = (40.0, 400.0)
LAMBDA_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)

RESULT_FIELDS = [
    "run_date", "run_ts", "method_tag", "status", "encoder", "head", "fill",
    "scope", "horizon", "n", "rmse", "mard",
    "dts_a", "dts_b", "dts_c", "dts_d", "dts_e",
    "fit_windows", "eval_rows", "fallback_rows", "ridge_lambda", "runtime_s", "notes",
]


def features(emb):
    """(N, 24, D) patch embeddings -> (N, 2D): last hour, then whole day."""
    return np.concatenate([emb[:, -1, :], emb.mean(axis=1)], axis=1).astype(np.float64)


def embed_features(model, hist, batch_size, device):
    """
    features(embed_windows(...)), one forward batch at a time.

    The (N, 24, D) patch embeddings are 6x larger than the features taken from
    them and are never needed whole, so reducing each batch as it comes off the
    encoder keeps them from ever existing at chunk size. Batch boundaries are
    unchanged, so every window meets the encoder in the same batch as before.
    """
    out = np.empty((len(hist), 2 * model.embed_dim), np.float64)
    for i in range(0, len(hist), batch_size):
        out[i : i + batch_size] = features(embed_windows(model, hist[i : i + batch_size], batch_size, device))
    return out


def peak_rss_gb():
    """Process high-water mark. ru_maxrss is bytes on macOS and KiB on Linux."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r * 1024 / 1e9


def key_table(rows):
    """rows' key columns as an Arrow table of plain strings (categoricals decoded)."""
    cols = {}
    for c in TEMPLATE_KEY:
        arr = pa.Array.from_pandas(rows[c])
        cols[c] = arr.dictionary_decode() if pa.types.is_dictionary(arr.type) else arr
    return pa.table(cols)


def load_truth(targets_path, rows):
    """
    Targets for `rows`, joined by key and returned in `rows` order.

    The join runs in Arrow: the full targets file as a pandas frame is ~430 MB of
    Python strings, which would make the scored phase heavier than the full pass.
    Arrow hashes the right-hand table, so the requested rows go on the right and
    the 2.65 M-row targets file is only probed, never hashed. The outer side is
    the requested rows, so a missing target surfaces as NaN for the caller's
    assertion rather than as a silently shorter frame.
    """
    keys = key_table(rows).append_column("_pos", pa.array(np.arange(len(rows), dtype=np.int64)))
    tgt = pq.read_table(targets_path, columns=TEMPLATE_KEY + TARGET_COLS)
    joined = tgt.join(keys, keys=TEMPLATE_KEY, join_type="right outer").sort_by("_pos")
    return joined.select(TEMPLATE_KEY + TARGET_COLS).to_pandas()


def write_predictions(path, rows, preds, template_path):
    """
    Key columns plus pred_*, cast to the template's own schema.

    Built in Arrow rather than pandas so categorical keys decode into compact
    Arrow strings instead of millions of Python objects. Casting to the template
    schema carries its pandas metadata too, so the file reads back with the same
    dtypes the template does, and a column type drifting from the submission
    contract fails here rather than at upload.
    """
    tbl = key_table(rows)
    for j, c in enumerate(PRED_COLS):
        tbl = tbl.append_column(c, pa.array(preds[:, j]))
    pq.write_table(tbl.cast(pq.read_schema(template_path)), path)


def fit_ridge(X, Y, groups, lambdas=LAMBDA_GRID, seed=0):
    """
    Ridge on standardized features with an intercept. The penalty is selected on a
    participant-disjoint holdout: 24 h windows from one participant overlap heavily,
    so a random split would tune lambda against near-copies of the fitting rows.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    held = set(rng.choice(uniq, size=max(1, len(uniq) // 5), replace=False))
    val = np.array([g in held for g in groups])
    tr = ~val
    if tr.sum() < 50 or val.sum() < 10:
        tr = np.ones(len(X), bool)
        val = tr

    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd
    y_mu = Y[tr].mean(0)
    A = Xs[tr].T @ Xs[tr]
    b = Xs[tr].T @ (Y[tr] - y_mu)
    eye = np.eye(A.shape[0])

    best = (np.inf, lambdas[0])
    for lam in lambdas:
        W = np.linalg.solve(A + lam * eye, b)
        err = (Xs[val] @ W + y_mu) - Y[val]
        rmse = float(np.sqrt((err ** 2).mean()))
        if rmse < best[0]:
            best = (rmse, lam)
    holdout_rmse, lam = best
    pers_rmse = float(np.sqrt((Y[val] ** 2).mean()))

    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd
    y_mu = Y.mean(0)
    W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(X.shape[1]), Xs.T @ (Y - y_mu))
    return {"mu": mu, "sd": sd, "y_mu": y_mu, "W": W, "lambda": lam,
            "holdout_rmse": holdout_rmse, "holdout_persistence_rmse": pers_rmse}


def apply_ridge(head, X, anchor):
    delta = ((X - head["mu"]) / head["sd"]) @ head["W"] + head["y_mu"]
    return np.clip(anchor[:, None] + delta, *GLUCOSE_RANGE)


def append_results(path, rows):
    """Append-only: a re-run adds rows, it never rewrites them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.parent / f".{path.name}.lock"
    lock.touch(exist_ok=True)
    with open(lock, "r+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            header = not path.exists() or path.stat().st_size == 0
            pd.DataFrame(rows, columns=RESULT_FIELDS).to_csv(path, mode="a", header=header, index=False)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def score(pred, true):
    z = calculate_dts_error_grid(pred, true)
    return dict(
        n=int(len(true)), rmse=calculate_rmse(pred, true), mard=calculate_mard(pred, true),
        dts_a=z["DTS_A_ZONE_PERCENT"], dts_b=z["DTS_B_ZONE_PERCENT"],
        dts_c=z["DTS_C_ZONE_PERCENT"], dts_d=z["DTS_D_ZONE_PERCENT"], dts_e=z["DTS_E_ZONE_PERCENT"],
    )


def report(name, per_horizon):
    print(f"\n  {name}")
    print(f"    {'horizon':>8} {'RMSE':>8} {'MARD %':>8} {'A %':>7} {'B %':>7} {'C+D+E %':>8}")
    for h, m in per_horizon.items():
        print(f"    {h:>8} {m['rmse']:>8.2f} {m['mard']:>8.2f} {m['dts_a']:>7.1f} "
              f"{m['dts_b']:>7.1f} {m['dts_c'] + m['dts_d'] + m['dts_e']:>8.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="Output partition, YYYY-MM-DD (never defaulted)")
    ap.add_argument("--scope", default="sub", choices=["sub", "full"],
                    help="sub: scored subsample (not submittable). full: template-complete parquet")
    ap.add_argument("--competition", default="live", choices=["live", "annual"])
    ap.add_argument("--encoder", default="x_cgm_jepa", choices=["x_cgm_jepa", "cgm_jepa"])
    ap.add_argument("--weights", default="weights/cgm_jepa_hf")
    ap.add_argument("--data-dir", default=None, help="overrides $DATA_DIR / data")
    ap.add_argument("--eval-rows", type=int, default=20000, help="template rows to score in --scope sub")
    ap.add_argument("--sample-mode", default="proportional", choices=["proportional", "balanced"])
    ap.add_argument("--only-source", default=None,
                    help="restrict to these source_file datasets (space or comma separated). "
                         "A shard flag: each shard writes its own results row and parquet, "
                         "so a --scope full shard is a PARTIAL submission and must be "
                         "assembled before run.py will accept it")
    ap.add_argument("--eval-chunk", type=int, default=EVAL_CHUNK,
                    help="windows embedded per batch; sets peak memory, not the result")
    ap.add_argument("--fit-row-groups", type=int, default=8, help="train.parquet row groups to read")
    ap.add_argument("--fit-per-group", type=int, default=150, help="max fit windows per participant")
    ap.add_argument("--fill", default="interp", choices=["interp", "sentinel"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="auto uses a GPU when one is visible. The encoder is small enough "
                         "that a GPU mainly buys queue time on a busy cluster, not throughput")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="windows per forward pass (default: 512 on CPU, 4096 on GPU)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        sys.exit(f"--date must be YYYY-MM-DD, got {args.date!r}")

    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir or os.environ.get("DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    comp_dir = data_dir / ("live_leaderboard" if args.competition == "live" else "annual_competition")

    needed = [data_dir / "train.parquet", data_dir / "test.parquet", comp_dir / "template.parquet"]
    if args.scope == "sub":
        needed.append(comp_dir / "targets.parquet")
    missing = [p for p in needed if not p.exists()]
    if missing:
        sys.exit("Missing required inputs:\n  " + "\n  ".join(str(p) for p in missing))

    out_dir = root / "results" / args.date / EXPERIMENT
    log_dir = root / "logs" / args.date / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet
    t_start = time.time()
    rng = np.random.default_rng(args.seed)

    print(f"CGM-JEPA zero-shot-encoder baseline | encoder={args.encoder} scope={args.scope} "
          f"competition={args.competition} date={args.date}")

    print("\n[1/5] Loading frozen encoder")
    device = resolve_device(args.device)
    model, cfg = load_encoder(root / args.weights, args.encoder, device=device)
    n_par = sum(p.numel() for p in model.parameters())
    batch_size = args.batch_size or (4096 if device.type == "cuda" else 512)
    print(f"    {args.encoder}: {n_par:,} parameters, embed_dim={cfg['embed_dim']}, frozen")
    print(f"    device={device} batch_size={batch_size}")
    if device.type == "cuda":
        print(f"    gpu={torch.cuda.get_device_name(0)}")
        drift = check_device_parity(model, device)
        model.to(device)
        print(f"    CPU/GPU parity check passed (max |diff| {drift:.2e})")

    print("\n[2/5] Building fit windows from train.parquet")
    Hf, Yf, Sf = build_fit_set(data_dir / "train.parquet", args.fit_row_groups,
                               args.fit_per_group, fill=args.fill, rng=rng, verbose=verbose)
    anchor_f = Hf[:, -1].astype(np.float64)
    print(f"    {len(Hf):,} fit windows across {len(np.unique(Sf))} source datasets: "
          f"{', '.join(sorted(np.unique(Sf)))}")

    print("\n[3/5] Embedding fit windows and solving the ridge head")
    Xf = embed_features(model, Hf, batch_size, device)
    head = fit_ridge(Xf, Yf - anchor_f[:, None], Sf, seed=args.seed)
    print(f"    lambda={head['lambda']:g}, {Xf.shape[1]} features")
    print(f"    participant-disjoint holdout delta-RMSE: ridge {head['holdout_rmse']:.2f} "
          f"vs persistence {head['holdout_persistence_rmse']:.2f} mg/dL")
    n_fit = len(Hf)
    del Hf, Yf, Sf, Xf, anchor_f

    print("\n[4/5] Selecting the evaluation rows")
    template = read_template(comp_dir / "template.parquet")
    rows = (template if args.scope == "full"
            else sample_template_rows(template, args.eval_rows, args.sample_mode, args.seed))
    if args.only_source:
        keep = {s for s in re.split(r"[,\s]+", args.only_source) if s}
        unknown = keep - set(template["source_file"].unique())
        if unknown:
            sys.exit(f"--only-source names datasets absent from the template: {sorted(unknown)}")
        rows = rows[rows["source_file"].isin(keep)]
        if rows.empty:
            sys.exit(f"--only-source {sorted(keep)} selected no rows")
    rows = rows.reset_index(drop=True)
    print(f"    {len(rows):,} of {len(template):,} template rows selected"
          + (f" (shard: {args.only_source})" if args.only_source else ""))

    print("\n[5/5] Windowing and predicting in chunks of "
          f"{args.eval_chunk:,} (peak memory is set by this, not by row count)")
    preds = np.full((len(rows), len(HORIZONS)), np.nan)
    anchor_e = np.full(len(rows), np.nan)
    n_windowed = 0
    for idx, He in iter_eval_chunks(data_dir / "test.parquet", rows, fill=args.fill,
                                    chunk=args.eval_chunk, verbose=verbose):
        anchor = He[:, -1].astype(np.float64)
        anchor_e[idx] = anchor
        Xe = embed_features(model, He, batch_size, device)
        preds[idx] = apply_ridge(head, Xe, anchor)
        n_windowed += len(idx)
        del He, Xe
    n_fallback = len(rows) - n_windowed
    print(f"    {n_windowed:,} windowed, {n_fallback:,} without usable 24 h history")

    gap = ~np.isfinite(anchor_e)
    if gap.any():
        anchor_e[gap] = np.nanmedian(anchor_e) if np.isfinite(anchor_e).any() else 120.0
        preds[gap] = anchor_e[gap, None]
    persistence = np.repeat(anchor_e[:, None], len(HORIZONS), axis=1)

    tag_base = f"jepa_{args.encoder}_ridge_{args.fill}"
    # The shard goes in the tag so concurrent shards never share a results CSV or
    # a predictions parquet, and so a partial file is never mistaken for a whole one.
    shard = "_" + re.sub(r"[^A-Za-z0-9]+", "-", args.only_source).strip("-") if args.only_source else ""
    scope_tag = f"{args.competition}{args.scope}{len(rows)}{shard}"
    runtime = time.time() - t_start
    rows_out = []
    common = dict(fill=args.fill, scope=scope_tag, fit_windows=n_fit, eval_rows=len(rows),
                  fallback_rows=int(n_fallback), runtime_s=round(runtime, 1))

    if args.scope == "sub" and args.competition == "live":
        truth = load_truth(comp_dir / "targets.parquet", rows)
        assert len(truth) == len(rows) and truth[TARGET_COLS].notna().all().all(), \
            "target join produced missing or duplicated rows"

        jepa_m, pers_m = {}, {}
        for j, h in enumerate(HORIZONS):
            y = truth[f"target_{h}"].to_numpy()
            jepa_m[h] = score(preds[:, j], y)
            pers_m[h] = score(persistence[:, j], y)
        report("CGM-JEPA + ridge (frozen encoder)", jepa_m)
        report("Persistence (last CGM held flat)", pers_m)
        print("\n    RMSE vs persistence: " + "  ".join(
            f"{h}m {jepa_m[h]['rmse'] - pers_m[h]['rmse']:+.2f}" for h in HORIZONS)
            + "   (negative = JEPA better)")

        for is_jepa, tag, mets in ((True, tag_base, jepa_m), (False, "persistence", pers_m)):
            for h in HORIZONS:
                rows_out.append(dict(
                    run_date=args.date, run_ts=datetime.now().isoformat(timespec="seconds"),
                    method_tag=f"{tag}_h{h}_{scope_tag}", status="ok",
                    encoder=args.encoder if is_jepa else "-", head="ridge" if is_jepa else "none",
                    horizon=h, **mets[h], **common,
                    ridge_lambda=head["lambda"] if is_jepa else "",
                    notes="frozen encoder, delta-from-current target" if is_jepa
                          else "reference baseline on identical rows",
                ))
    else:
        for h in HORIZONS:
            rows_out.append(dict(
                run_date=args.date, run_ts=datetime.now().isoformat(timespec="seconds"),
                method_tag=f"{tag_base}_h{h}_{scope_tag}", status="ok",
                encoder=args.encoder, head="ridge", horizon=h, n=len(rows),
                rmse="", mard="", dts_a="", dts_b="", dts_c="", dts_d="", dts_e="",
                **common, ridge_lambda=head["lambda"],
                notes="predictions written; scored by run.py or server-side",
            ))

    pred_dir = out_dir / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{tag_base}_{scope_tag}.parquet"
    write_predictions(pred_path, rows, preds, comp_dir / "template.parquet")

    results_csv = out_dir / f"{EXPERIMENT}_{scope_tag}_results.csv"
    append_results(results_csv, rows_out)
    cfg_path = log_dir / f"{tag_base}_{scope_tag}.json"
    cfg_path.write_text(json.dumps(
        {**vars(args), "fit_windows": int(n_fit), "eval_rows": int(len(rows)),
         "fallback_rows": int(n_fallback), "ridge_lambda": float(head["lambda"]),
         "holdout_rmse": round(head["holdout_rmse"], 3),
         "holdout_persistence_rmse": round(head["holdout_persistence_rmse"], 3),
         "encoder_params": int(n_par), "device": str(device), "batch_size": batch_size,
         "runtime_s": round(runtime, 1), "peak_rss_gb": round(peak_rss_gb(), 2)}, indent=2), encoding="utf-8")

    print(f"\n  predictions -> {pred_path.relative_to(root)}")
    print(f"  results     -> {results_csv.relative_to(root)}")
    print(f"  run config  -> {cfg_path.relative_to(root)}")
    print(f"  runtime     -> {runtime:.1f}s")
    print(f"  peak RSS    -> {peak_rss_gb():.2f} GB")
    if args.scope == "full" and args.only_source:
        print(f"\n  PARTIAL: this shard holds {len(rows):,} of {len(template):,} template rows. "
              f"Assemble every shard before validating:\n"
              f"    python assemble_submission.py --date {args.date} --competition {args.competition}")
    elif args.scope == "full":
        print(f"\n  validate with:\n    python run.py {pred_path.relative_to(root)} "
              f"--competition {args.competition} --horizon all")
    else:
        print("\n  NOTE: a subsample has fewer rows than the template, so run.py rejects it by "
              "design.\n        Use --scope full for a submittable parquet.")


if __name__ == "__main__":
    main()
