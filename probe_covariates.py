#!/usr/bin/env python3
"""
probe_covariates.py

How much would insulin and carbs add to the frozen CGM-JEPA baseline? A scored
diagnostic on the live subsample -- not a submission path.

Every variant is fitted on the same train.parquet windows and scored on the same
template rows that predict_jepa.py --scope sub uses, against the same persistence
row, so the rows of the results file differ only by the inputs to the head:

    jepa         the 192 frozen-encoder features alone (reproduces predict_jepa)
    jepahist     + insulin/carb history up to t
    jepahistfut  + known insulin/carbs after t, up to the forecast horizon
    cgmhistfut   the last 2 h of raw CGM instead of JEPA features, same covariates

    rich         JEPA + CGM summaries + the covariates at 5-min resolution, their
                 expected action inside the horizon, and time of day
    richnojepa   rich without the JEPA features

cgmhistfut and richnojepa are the controls: if they match their JEPA twins, the
encoder is adding nothing once the covariates are present.

Covariates are windowed sums on the participant's 5-minute grid, per channel
(bolus U, basal U, carbs g; missing counted as zero), plus each channel's
observed fraction over the past 6 h so "no carbs logged" and "carbs not recorded"
are separable:

    history   (t-30, t], (t-60, t-30], (t-120, t-60], (t-240, t-120], (t-360, t-240]
    future    (t, t+30], (t+30, t+60], (t+60, t+90], (t+90, t+120]

A horizon-h head sees only the future bins that end by t+h: insulin or carbs after
the target time cannot move it, and in closed-loop data it is partly the
controller reacting to the glucose being predicted. Even inside the horizon,
automated basal is a response to CGM after t, so jepahistfut is an upper bound
for any dataset run by an AID system; the rules allow it, but read it as such.

The rich block adds, per channel, the last hour and next 2 h as raw 5-min
slots, and each dose's expected effect inside (t, t+h] under a first-order
exponential response (two time constants per channel); see TAUS.

Heads: closed-form ridge (predict_jepa.fit_ridge, per horizon), and with --gbm a
histogram gradient-boosted tree per horizon (scikit-learn). Both predict the
change from G(t), as the baseline does.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from covariates import RAW_CGM, Covariates, cgm_features
from jepa_model import load_encoder, resolve_device
from jepa_windows import (
    HORIZONS, build_fit_set, iter_eval_chunks, read_template, sample_template_rows,
)
from predict_jepa import (
    GLUCOSE_RANGE, TARGET_COLS, append_results, embed_features, fit_ridge, load_truth,
    peak_rss_gb, score,
)

EXPERIMENT = "jepa_covariates"
HORIZON_FREE = ("jepa", "jepahist")   # variants whose features do not depend on h
GBM_VARIANTS = ("jepahistfut", "cgmhistfut", "rich", "richnojepa")
ALL_VARIANTS = ("jepa", "jepahist", "jepahistfut", "cgmhistfut", "rich", "richnojepa")


class GBMHead:
    """One histogram gradient-boosted tree per horizon on the change from G(t)."""

    def __init__(self, seed, max_iter=500):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.label = "gbm" if max_iter == 500 else f"gbm{max_iter}"
        self.make = lambda: HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            random_state=seed)

    def fit_predict(self, X, y, Xe):
        m = self.make().fit(X, y)
        return m.predict(Xe)


def ridge_fit_predict(X, y, groups, Xe, seed):
    head = fit_ridge(X, y[:, None], groups, seed=seed)
    return (((Xe - head["mu"]) / head["sd"]) @ head["W"] + head["y_mu"])[:, 0], head["lambda"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="Output partition, YYYY-MM-DD (never defaulted)")
    ap.add_argument("--encoder", default="x_cgm_jepa", choices=["x_cgm_jepa", "cgm_jepa"])
    ap.add_argument("--fill", default="interp", choices=["interp", "sentinel"])
    ap.add_argument("--weights", default="weights/cgm_jepa_hf")
    ap.add_argument("--data-dir", default=None, help="overrides $DATA_DIR / data")
    ap.add_argument("--eval-rows", type=int, default=200000)
    ap.add_argument("--eval-chunk", type=int, default=10000)
    ap.add_argument("--fit-row-groups", type=int, default=24)
    ap.add_argument("--fit-per-group", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--gbm", action="store_true", help="also fit gradient-boosted heads (needs scikit-learn)")
    ap.add_argument("--gbm-iters", type=int, default=500,
                    help="boosting rounds cap (early stopping still applies); a non-default "
                         "value is carried in the head label, e.g. gbm2000")
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS),
                    help=f"comma-separated subset of {','.join(ALL_VARIANTS)}")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        sys.exit(f"--date must be YYYY-MM-DD, got {args.date!r}")
    chosen = [v for v in args.variants.split(",") if v]
    unknown = set(chosen) - set(ALL_VARIANTS)
    if unknown:
        sys.exit(f"--variants: unknown {sorted(unknown)}; choose from {ALL_VARIANTS}")
    gbm = None
    if args.gbm:
        try:
            gbm = GBMHead(args.seed, args.gbm_iters)
        except ImportError:
            sys.exit("--gbm needs scikit-learn:  python3 -m pip install scikit-learn")

    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir or os.environ.get("DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    comp_dir = data_dir / "live_leaderboard"
    needed = [data_dir / "train.parquet", data_dir / "test.parquet",
              comp_dir / "template.parquet", comp_dir / "targets.parquet"]
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
    cov = Covariates()
    print(f"Covariate probe | encoder={args.encoder} fill={args.fill} date={args.date} gbm={bool(gbm)}")

    print("\n[1/5] Loading frozen encoder")
    device = resolve_device("cpu")
    model, _ = load_encoder(root / args.weights, args.encoder, device=device)

    print("\n[2/5] Fit windows + covariates from train.parquet")
    Hf, Yf, Sf, Cf = build_fit_set(data_dir / "train.parquet", args.fit_row_groups, args.fit_per_group,
                                   fill=args.fill, rng=rng, verbose=verbose, extra=cov)
    anchor_f = Hf[:, -1].astype(np.float64)
    Jf = embed_features(model, Hf, args.batch_size, device)
    Rf = Hf[:, -RAW_CGM:].astype(np.float64)
    Gf = cgm_features(Hf)
    Df = Yf - anchor_f[:, None]
    n_fit = len(Hf)
    del Hf
    print(f"    {n_fit:,} fit windows, {cov.n} covariate columns")

    print("\n[3/5] Evaluation rows (same sample as predict_jepa --scope sub)")
    template = read_template(comp_dir / "template.parquet")
    rows = sample_template_rows(template, args.eval_rows, "proportional", args.seed).reset_index(drop=True)
    n = len(rows)
    Je = np.zeros((n, Jf.shape[1]))
    Re = np.zeros((n, RAW_CGM))
    Ge = np.zeros((n, Gf.shape[1]))
    Ce = np.zeros((n, cov.n))
    anchor_e = np.full(n, np.nan)
    for idx, He, Xc in iter_eval_chunks(data_dir / "test.parquet", rows, fill=args.fill,
                                        chunk=args.eval_chunk, verbose=verbose, extra=cov):
        anchor_e[idx] = He[:, -1]
        Je[idx] = embed_features(model, He, args.batch_size, device)
        Re[idx] = He[:, -RAW_CGM:]
        Ge[idx] = cgm_features(He)
        Ce[idx] = Xc
    windowed = np.isfinite(anchor_e)
    # Same fallback as predict_jepa, so the jepa row reproduces it exactly.
    anchor_e[~windowed] = np.nanmedian(anchor_e)
    print(f"    {n:,} rows, {windowed.sum():,} windowed")

    truth = load_truth(comp_dir / "targets.parquet", rows)
    assert len(truth) == n and truth[TARGET_COLS].notna().all().all(), "target join failed"
    Y = truth[TARGET_COLS].to_numpy(np.float64)

    variants = {
        "jepa":        lambda h: (Jf, Je),
        "jepahist":    lambda h: (np.hstack([Jf, Cf[:, cov.hist_cols]]), np.hstack([Je, Ce[:, cov.hist_cols]])),
        "jepahistfut": lambda h: (np.hstack([Jf, Cf[:, cov.hist_cols], Cf[:, cov.fut_cols(h)]]),
                                  np.hstack([Je, Ce[:, cov.hist_cols], Ce[:, cov.fut_cols(h)]])),
        "cgmhistfut":  lambda h: (np.hstack([Rf, Cf[:, cov.hist_cols], Cf[:, cov.fut_cols(h)]]),
                                  np.hstack([Re, Ce[:, cov.hist_cols], Ce[:, cov.fut_cols(h)]])),
        "rich":        lambda h: (np.hstack([Jf, Gf, Cf[:, cov.rich_cols(h)]]),
                                  np.hstack([Je, Ge, Ce[:, cov.rich_cols(h)]])),
        "richnojepa":  lambda h: (np.hstack([Gf, Cf[:, cov.rich_cols(h)]]),
                                  np.hstack([Ge, Ce[:, cov.rich_cols(h)]])),
    }
    variants = {k: v for k, v in variants.items() if k in chosen}
    heads = ["ridge"] + ([gbm.label] if gbm else [])

    print("\n[4/5] Fitting heads and scoring")
    preds, lambdas = {}, {}
    for head in heads:
        for name, feats in variants.items():
            if head != "ridge" and name not in GBM_VARIANTS:
                continue          # the question is what covariates add at full strength
            P = np.empty((n, len(HORIZONS)))
            if head == "ridge" and name in HORIZON_FREE:
                # Same features at every horizon: one joint fit with a shared
                # penalty, exactly as predict_jepa does, so the jepa row
                # reproduces its subsample numbers.
                Xtr, Xev = feats(None)
                r = fit_ridge(Xtr, Df, Sf, seed=args.seed)
                D = ((Xev - r["mu"]) / r["sd"]) @ r["W"] + r["y_mu"]
                for j, h in enumerate(HORIZONS):
                    lambdas[(name, h)] = r["lambda"]
                    P[:, j] = np.clip(anchor_e + D[:, j], *GLUCOSE_RANGE)
            else:
                for j, h in enumerate(HORIZONS):
                    Xtr, Xev = feats(h)
                    if head == "ridge":
                        d, lambdas[(name, h)] = ridge_fit_predict(Xtr, Df[:, j], Sf, Xev, args.seed)
                    else:
                        d = gbm.fit_predict(Xtr, Df[:, j], Xev)
                    P[:, j] = np.clip(anchor_e + d, *GLUCOSE_RANGE)
            P[~windowed] = anchor_e[~windowed, None]
            preds[(name, head)] = P
            if verbose:
                print(f"    {name:12s} {head:7s} done ({time.time() - t_start:.0f}s)", flush=True)
    preds[("persistence", "none")] = np.repeat(anchor_e[:, None], len(HORIZONS), axis=1)

    print("\n[5/5] Results -- RMSE mg/dL on identical rows (overall = pooled over horizons)")
    print(f"    {'variant':12s} {'head':7s} " + " ".join(f"{h:>7d}" for h in HORIZONS) + "  overall")
    scope_tag = f"livesub{n}"
    now = datetime.now().isoformat(timespec="seconds")
    runtime = round(time.time() - t_start, 1)
    rows_out, summary = [], {}
    for (name, head), P in preds.items():
        per_h = {h: score(P[:, j], Y[:, j]) for j, h in enumerate(HORIZONS)}
        overall = float(np.sqrt(np.mean((P - Y) ** 2)))
        summary[f"{name}/{head}"] = {**{h: round(per_h[h]["rmse"], 2) for h in HORIZONS},
                                     "overall": round(overall, 2)}
        print(f"    {name:12s} {head:7s} " + " ".join(f"{per_h[h]['rmse']:7.2f}" for h in HORIZONS)
              + f"  {overall:7.2f}")
        tag = ("persistence" if name == "persistence"
               else f"jepacov_{args.encoder}_{name}_{head}_{args.fill}")
        for h in HORIZONS:
            rows_out.append(dict(
                run_date=args.date, run_ts=now, method_tag=f"{tag}_h{h}_{scope_tag}", status="ok",
                encoder="-" if name in ("persistence", "cgmhistfut", "richnojepa") else args.encoder,
                head=head, fill=args.fill, scope=scope_tag, horizon=h, **per_h[h],
                fit_windows=n_fit, eval_rows=n, fallback_rows=int((~windowed).sum()),
                ridge_lambda=lambdas.get((name, h), "") if head == "ridge" else "",
                runtime_s=runtime,
                notes={"jepa": "frozen JEPA features only",
                       "jepahist": "JEPA + insulin/carb history",
                       "jepahistfut": "JEPA + history + known future insulin/carbs to t+h (AID upper bound)",
                       "cgmhistfut": "last 2 h raw CGM + same covariates, no JEPA (control)",
                       "rich": "JEPA + CGM summaries + covariate bins, 5-min slots, dose-action features, time of day",
                       "richnojepa": "rich without JEPA features (control)",
                       "persistence": "reference baseline on identical rows"}[name],
            ))

    results_csv = out_dir / f"{EXPERIMENT}_{scope_tag}_results.csv"
    append_results(results_csv, rows_out)
    cfg_path = log_dir / f"jepacov_{args.encoder}_{args.fill}_{scope_tag}.json"
    cfg_path.write_text(json.dumps(
        {**vars(args), "fit_windows": int(n_fit), "eval_rows": int(n),
         "fallback_rows": int((~windowed).sum()), "covariate_columns": int(cov.n),
         "rmse": summary, "runtime_s": runtime, "peak_rss_gb": round(peak_rss_gb(), 2)},
        indent=2, default=str), encoding="utf-8")
    print(f"\n  results    -> {results_csv.relative_to(root)}")
    print(f"  run config -> {cfg_path.relative_to(root)}")
    print(f"  runtime    -> {runtime:.1f}s, peak RSS {peak_rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
