#!/usr/bin/env python3
"""
finetune_jepa.py

Fine-tune the CGM-JEPA encoder end to end as the CGM branch of a multi-horizon
forecaster that also reads insulin and carbohydrates, including the known future
doses the competition allows.

    CGM history (288 x 5 min, raw mg/dL)
        -> CGM-JEPA encoder (published weights, or random, or frozen)
        -> last-hour token + mean token                     (2 x 96)
    CGM summaries (last 2 h, recent changes, spread, mean)   (30)
    covariates.Covariates block, every column                (188)
        -> signed log1p, standardised on the training split
        -> MLP                                               (256)
    concat -> MLP head -> change from G(t) at 30/60/90/120 min

Every horizon sees every covariate column, including future doses past t+h:
the competition hands the model the whole forecast window's insulin and carbs,
and the target here is the with-future score.

The loss is MSE on the four changes (scaled by 1/50) with equal weight, i.e. the
pooled "overall" RMSE the leaderboard reports.

Three controls share everything but the encoder:
    --init pretrained   published weights, fine-tuned (the experiment)
    --init scratch      same architecture, random init (does pretraining help?)
    --init frozen       published weights, never updated (does fine-tuning help?)
and --no-covariates drops the covariate branch.

Windows are built once through jepa_windows (the same 288-sample grid and
validity rule as every other method here) and cached as .npy under
cache/finetune_windows/, keyed by the sampling arguments. Validation holds out
whole participants.

Outputs:
  checkpoints/<date>/jepa_finetune/<tag>.pt          best-validation weights + normaliser
  logs/<date>/jepa_finetune/<tag>_epochs.csv         per-epoch train/val curves
  logs/<date>/jepa_finetune/<tag>_<scope>.json       run config
  results/<date>/jepa_finetune/jepa_finetune_livesub<n>_results.csv
      the 200 k subsample predict_jepa and probe_covariates score, with persistence
  results/<date>/jepa_finetune/preds/<tag>_livefull<n>.parquet   with --predict-full
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from covariates import RAW_CGM, Covariates
from jepa_model import NUM_PATCHES, PATCH_LEN, CGMJepaEncoder, load_encoder, resolve_device
from jepa_windows import (
    HORIZONS, build_fit_set, iter_eval_chunks, read_template, sample_template_rows,
)
from predict_jepa import (
    GLUCOSE_RANGE, TARGET_COLS, append_results, load_truth, peak_rss_gb, score,
    write_predictions,
)

EXPERIMENT = "jepa_finetune"
CACHE_DIR = Path("cache") / "finetune_windows"
DELTA_SCALE = 50.0              # targets and outputs are (G(t+h) - G(t)) / 50
G_CENTER, G_SCALE = 140.0, 60.0
N_CGM_SUMMARY = RAW_CGM + 4 + 1 + 1


# --------------------------------------------------------------------------
# Window cache
# --------------------------------------------------------------------------

class ParticipantTagged:
    """
    Wraps the Covariates hook and appends an integer participant code as a last
    column, so validation can hold out whole participants. Codes are assigned in
    first-seen order and are only meaningful within one cache.
    """

    def __init__(self, cov):
        self.cov = cov
        self.columns = cov.columns
        self.codes = {}

    def __call__(self, frame, origins_ns):
        key = (str(frame["source_file"].iloc[0]), str(frame["id"].iloc[0]))
        code = self.codes.setdefault(key, len(self.codes))
        x = self.cov(frame, origins_ns)
        return np.hstack([x, np.full((len(x), 1), code, np.float64)])


def _save(d, **arrays):
    d.mkdir(parents=True, exist_ok=True)
    for name, a in arrays.items():
        np.save(d / f"{name}.npy", a)


def train_cache(root, data_dir, args, verbose):
    """(hist, targets, covariates, participant codes, sources); built once per key."""
    key = f"train_rg{args.fit_row_groups}_pg{args.fit_per_group}_{args.fill}_s{args.seed}"
    d = root / CACHE_DIR / key
    names = ("hist", "target", "cov", "pid", "src")
    if all((d / f"{n}.npy").exists() for n in names):
        print(f"    cached: {d.relative_to(root)}")
        return tuple(np.load(d / f"{n}.npy", allow_pickle=(n == "src")) for n in names)
    print(f"    building {d.relative_to(root)} (once; later runs load it)")
    hook = ParticipantTagged(Covariates())
    H, Y, S, X = build_fit_set(data_dir / "train.parquet", args.fit_row_groups, args.fit_per_group,
                               fill=args.fill, rng=np.random.default_rng(args.seed),
                               verbose=verbose, extra=hook)
    out = (H.astype(np.float32), Y.astype(np.float32), X[:, :-1].astype(np.float32),
           X[:, -1].astype(np.int32), S.astype(str))
    _save(d, **dict(zip(names, out)))
    return out


def eval_cache(root, data_dir, comp_dir, rows, args, verbose):
    """Windows for the scored subsample: (row positions, hist, covariates)."""
    d = root / CACHE_DIR / f"eval_sub{len(rows)}_{args.fill}_s{args.seed}"
    names = ("idx", "hist", "cov")
    if all((d / f"{n}.npy").exists() for n in names):
        print(f"    cached: {d.relative_to(root)}")
        return tuple(np.load(d / f"{n}.npy") for n in names)
    I, H, C = [], [], []
    for idx, he, xc in iter_eval_chunks(data_dir / "test.parquet", rows, fill=args.fill,
                                        chunk=args.eval_chunk, verbose=verbose, extra=Covariates()):
        I.append(idx)
        H.append(he.astype(np.float32))
        C.append(xc.astype(np.float32))
    out = (np.concatenate(I), np.concatenate(H), np.concatenate(C))
    _save(d, **dict(zip(names, out)))
    return out


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class CovNorm:
    """Signed log1p on dose-like columns, then standardisation fitted on training rows."""

    def __init__(self, n_cols, passthrough):
        self.log = np.ones(n_cols, bool)
        self.log[passthrough] = False
        self.mu = np.zeros(n_cols, np.float32)
        self.sd = np.ones(n_cols, np.float32)

    def _pre(self, c):
        c = c.astype(np.float32, copy=True)
        c[:, self.log] = np.sign(c[:, self.log]) * np.log1p(np.abs(c[:, self.log]))
        return c

    def fit(self, c):
        z = self._pre(c)
        self.mu = z.mean(0)
        self.sd = z.std(0)
        self.sd[self.sd < 1e-6] = 1.0
        return self

    def __call__(self, c):
        return (self._pre(c) - self.mu) / self.sd

    def state(self):
        return {"log": self.log, "mu": self.mu, "sd": self.sd}

    @classmethod
    def from_state(cls, st):
        n = cls(len(st["mu"]), [])
        n.log, n.mu, n.sd = np.asarray(st["log"]), np.asarray(st["mu"]), np.asarray(st["sd"])
        return n


def cgm_summary(h):
    """Torch twin of covariates.cgm_features, scaled for a network."""
    g = (h - G_CENTER) / G_SCALE
    deltas = (h[:, -1:] - h[:, [-2, -4, -7, -13]]) / 30.0
    return torch.cat([g[:, -RAW_CGM:], deltas, h[:, -12:].std(dim=1, keepdim=True) / 30.0,
                      g.mean(dim=1, keepdim=True)], dim=1)


class Forecaster(nn.Module):
    def __init__(self, encoder, n_cov, use_cov=True, hidden=512, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.embed_dim
        self.use_cov = use_cov
        cov_dim = 256 if use_cov else 0
        self.cov_mlp = (nn.Sequential(nn.Linear(n_cov, 256), nn.GELU(), nn.Linear(256, cov_dim))
                        if use_cov else None)
        n_in = 2 * d + N_CGM_SUMMARY + cov_dim
        self.head = nn.Sequential(
            nn.LayerNorm(n_in), nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, len(HORIZONS)))

    def forward(self, hist, cov):
        tok = self.encoder(hist.view(-1, NUM_PATCHES, PATCH_LEN))
        z = [tok[:, -1], tok.mean(dim=1), cgm_summary(hist)]
        if self.use_cov:
            z.append(self.cov_mlp(cov))
        return self.head(torch.cat(z, dim=1))


def pick_device(spec):
    """cuda > mps > cpu for auto. cuda/cpu go through jepa_model (TF32 pinned off)."""
    if spec == "mps" or (spec == "auto" and not torch.cuda.is_available()
                         and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return torch.device("mps")
    return resolve_device(spec)


def build_model(root, args, n_cov):
    if args.init == "scratch":
        cfg = json.loads((root / args.weights / args.encoder / "config.json").read_text())
        encoder = CGMJepaEncoder(**cfg)
    else:
        encoder, _ = load_encoder(root / args.weights, args.encoder, device="cpu")
    # proj is carried only so published state dicts load; it is on no forward path.
    for p in encoder.proj.parameters():
        p.requires_grad_(False)
    if args.init == "frozen":
        for p in encoder.parameters():
            p.requires_grad_(False)
    return Forecaster(encoder, n_cov, use_cov=not args.no_covariates,
                      hidden=args.hidden, dropout=args.dropout)


@torch.no_grad()
def predict_deltas(model, H, C, device, batch=4096):
    """(N, 4) predicted change from G(t), mg/dL."""
    model.eval()
    out = np.empty((len(H), len(HORIZONS)), np.float64)
    for i in range(0, len(H), batch):
        h = torch.from_numpy(np.ascontiguousarray(H[i : i + batch])).to(device)
        c = torch.from_numpy(np.ascontiguousarray(C[i : i + batch])).to(device)
        out[i : i + batch] = model(h, c).float().cpu().numpy() * DELTA_SCALE
    return out


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def participant_split(pid, frac, seed):
    uniq = np.unique(pid)
    rng = np.random.default_rng(seed)
    held = rng.choice(uniq, size=max(1, int(round(len(uniq) * frac))), replace=False)
    val = np.isin(pid, held)
    return np.flatnonzero(~val), np.flatnonzero(val)


def train(model, H, D, C, tr, va, device, args, log_path, ckpt_path, extra_state):
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    groups = [{"params": head_params, "lr": args.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr * args.encoder_lr_mult})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    base = [g["lr"] for g in opt.param_groups]

    steps_per_epoch = args.steps_per_epoch or math.ceil(len(tr) / args.batch_size)
    total = steps_per_epoch * args.epochs
    warm = max(1, int(total * args.warmup_frac))

    def lr_at(step):
        if step < warm:
            return step / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

    rng = np.random.default_rng(args.seed)
    best, best_epoch, bad, step = np.inf, -1, 0, 0
    D_va = D[va]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    with open(log_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_log:
            w.writerow(["run_ts", "epoch", "steps", "train_rmse", "val_rmse_30", "val_rmse_60",
                        "val_rmse_90", "val_rmse_120", "val_overall", "lr", "windows_per_s", "elapsed_s"])
        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            if args.init == "frozen":
                model.encoder.eval()
            order = rng.permutation(tr)
            sq, n_seen, t_ep = 0.0, 0, time.time()
            for b in range(steps_per_epoch):
                start = (b * args.batch_size) % len(order)
                sel = np.sort(order[start : start + args.batch_size])
                h = torch.from_numpy(H[sel]).to(device)
                c = torch.from_numpy(C[sel]).to(device)
                y = torch.from_numpy(D[sel] / DELTA_SCALE).to(device)
                for g, lr0 in zip(opt.param_groups, base):
                    g["lr"] = lr0 * lr_at(step)
                loss = ((model(h, c) - y) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
                sq += float(loss.detach()) * len(sel)
                n_seen += len(sel)
                if args.log_every and b % args.log_every == 0:
                    print(f"      epoch {epoch} step {b}/{steps_per_epoch} "
                          f"loss {math.sqrt(sq / n_seen) * DELTA_SCALE:.2f} mg/dL "
                          f"({n_seen / (time.time() - t_ep):,.0f} windows/s)", flush=True)
            train_rmse = math.sqrt(sq / max(1, n_seen)) * DELTA_SCALE
            P = predict_deltas(model, H[va], C[va], device, args.eval_batch)
            per_h = np.sqrt(((P - D_va) ** 2).mean(axis=0))
            overall = float(np.sqrt(((P - D_va) ** 2).mean()))
            rate = n_seen / (time.time() - t_ep)
            print(f"    epoch {epoch}: train {train_rmse:.2f} | val " +
                  " ".join(f"{v:.2f}" for v in per_h) + f" | overall {overall:.2f} "
                  f"| {rate:,.0f} windows/s", flush=True)
            w.writerow([datetime.now().isoformat(timespec="seconds"), epoch, step, round(train_rmse, 3),
                        *[round(float(v), 3) for v in per_h], round(overall, 3),
                        f"{opt.param_groups[0]['lr']:.2e}", round(rate), round(time.time() - t0, 1)])
            fh.flush()
            if overall < best - 1e-3:
                best, best_epoch, bad = overall, epoch, 0
                torch.save({"model": model.state_dict(), **extra_state,
                            "epoch": epoch, "val_overall": overall}, ckpt_path)
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"    early stop: no val gain for {bad} epochs")
                    break
    return best, best_epoch


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="Output partition, YYYY-MM-DD (never defaulted)")
    ap.add_argument("--init", default="pretrained", choices=["pretrained", "scratch", "frozen"])
    ap.add_argument("--encoder", default="x_cgm_jepa", choices=["x_cgm_jepa", "cgm_jepa"])
    ap.add_argument("--no-covariates", action="store_true", help="CGM-only control")
    ap.add_argument("--fill", default="interp", choices=["interp", "sentinel"])
    ap.add_argument("--weights", default="weights/cgm_jepa_hf")
    ap.add_argument("--data-dir", default=None, help="overrides $DATA_DIR / data")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    # data
    ap.add_argument("--fit-row-groups", type=int, default=128)
    ap.add_argument("--fit-per-group", type=int, default=300, help="max training windows per participant")
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of participants held out")
    ap.add_argument("--eval-rows", type=int, default=200000)
    ap.add_argument("--eval-chunk", type=int, default=10000)
    # optimisation
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--steps-per-epoch", type=int, default=0, help="0 = one pass over the training split")
    ap.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    ap.add_argument("--encoder-lr-mult", type=float, default=0.1, help="encoder lr = lr x this")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-batch", type=int, default=4096)
    ap.add_argument("--log-every", type=int, default=200, help="steps between progress lines; 0 = off")
    # outputs
    ap.add_argument("--checkpoint", default=None,
                    help="skip training and evaluate/predict from this checkpoint")
    ap.add_argument("--predict-full", action="store_true",
                    help="also write a template-complete submission parquet")
    ap.add_argument("--only-source", default=None,
                    help="restrict --predict-full to these sources (a PARTIAL shard)")
    ap.add_argument("--tag-suffix", default="", help="appended to the method tag for deliberate variants")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        sys.exit(f"--date must be YYYY-MM-DD, got {args.date!r}")
    if args.tag_suffix and not re.fullmatch(r"[A-Za-z0-9]+", args.tag_suffix):
        sys.exit("--tag-suffix must be alphanumeric")

    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir or os.environ.get("DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    comp_dir = data_dir / "live_leaderboard"
    needed = [data_dir / "test.parquet", comp_dir / "template.parquet", comp_dir / "targets.parquet"]
    if not args.checkpoint:
        needed.append(data_dir / "train.parquet")
    missing = [p for p in needed if not p.exists()]
    if missing:
        sys.exit("Missing required inputs:\n  " + "\n  ".join(str(p) for p in missing))

    ckpt_in = None
    if args.checkpoint:
        ckpt_in = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        for k in ("init", "encoder", "no_covariates", "fill", "hidden", "dropout", "tag_suffix"):
            setattr(args, k, ckpt_in["args"][k])

    tag = (f"ftjepa_{args.encoder}_{args.init}_{'nocov' if args.no_covariates else 'cov'}"
           f"_{args.fill}" + (f"_{args.tag_suffix}" if args.tag_suffix else ""))
    out_dir = root / "results" / args.date / EXPERIMENT
    log_dir = root / "logs" / args.date / EXPERIMENT
    ckpt_dir = root / "checkpoints" / args.date / EXPERIMENT
    for d in (out_dir, log_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    verbose = not args.quiet
    t_start = time.time()
    cov = Covariates()
    print(f"CGM-JEPA fine-tune | tag={tag} device={device} threads={torch.get_num_threads()} date={args.date}")

    # ---------------------------------------------------------------- train
    print("\n[1/4] Model")
    model = build_model(root, args, cov.n)
    if ckpt_in is not None:
        model.load_state_dict(ckpt_in["model"])
        norm = CovNorm.from_state(ckpt_in["cov_norm"])
    model.to(device)
    n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    init={args.init} covariates={not args.no_covariates} "
          f"trainable parameters={n_train_p:,}")

    n_fit = best_val = best_epoch = None
    if ckpt_in is None:
        print("\n[2/4] Training windows")
        H, Y, C, pid, src = train_cache(root, data_dir, args, verbose)
        D = (Y - H[:, -1:]).astype(np.float32)
        tr, va = participant_split(pid, args.val_frac, args.seed)
        norm = CovNorm(cov.n, cov.slices["tod"]).fit(C[tr])
        Cn = norm(C).astype(np.float32)
        del C
        n_fit = len(H)
        print(f"    {n_fit:,} windows, {len(np.unique(pid)):,} participants, "
              f"{len(np.unique(src))} sources | train {len(tr):,} / val {len(va):,} (participant-disjoint)")
        ckpt_path = ckpt_dir / f"{tag}.pt"
        print(f"\n    training -> {ckpt_path.relative_to(root)}")
        best_val, best_epoch = train(
            model, H, D, Cn, tr, va, device, args, log_dir / f"{tag}_epochs.csv", ckpt_path,
            {"cov_norm": norm.state(), "args": vars(args), "n_fit": n_fit})
        print(f"    best val overall {best_val:.2f} mg/dL at epoch {best_epoch}")
        del H, Y, D, Cn
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        model.to(device)
    else:
        n_fit, best_val, best_epoch = ckpt_in.get("n_fit"), ckpt_in.get("val_overall"), ckpt_in.get("epoch")
        print("\n[2/4] Training skipped: --checkpoint given")

    # ------------------------------------------------------- scored subsample
    print("\n[3/4] Scoring the live subsample (same rows as predict_jepa / probe_covariates)")
    template = read_template(comp_dir / "template.parquet")
    rows = sample_template_rows(template, args.eval_rows, "proportional", args.seed).reset_index(drop=True)
    n = len(rows)
    idx, He, Ce = eval_cache(root, data_dir, comp_dir, rows, args, verbose)
    anchor = np.full(n, np.nan)
    anchor[idx] = He[:, -1]
    fallback = np.nanmedian(anchor)          # same fallback as the other drivers
    P = np.repeat(np.where(np.isfinite(anchor), anchor, fallback)[:, None], len(HORIZONS), axis=1)
    pers = P.copy()
    P[idx] = np.clip(He[:, -1:].astype(np.float64)
                     + predict_deltas(model, He, norm(Ce).astype(np.float32), device, args.eval_batch),
                     *GLUCOSE_RANGE)
    truth = load_truth(comp_dir / "targets.parquet", rows)
    assert len(truth) == n and truth[TARGET_COLS].notna().all().all(), "target join failed"
    Yt = truth[TARGET_COLS].to_numpy(np.float64)

    scope_tag = f"livesub{n}"
    now = datetime.now().isoformat(timespec="seconds")
    runtime = round(time.time() - t_start, 1)
    rows_out, summary = [], {}
    print(f"    {'method':52s} " + " ".join(f"{h:>7d}" for h in HORIZONS) + "  overall")
    for name, M in ((tag, P), ("persistence", pers)):
        per_h = {h: score(M[:, j], Yt[:, j]) for j, h in enumerate(HORIZONS)}
        overall = float(np.sqrt(np.mean((M - Yt) ** 2)))
        summary[name] = {**{h: round(per_h[h]["rmse"], 2) for h in HORIZONS}, "overall": round(overall, 2)}
        print(f"    {name:52s} " + " ".join(f"{per_h[h]['rmse']:7.2f}" for h in HORIZONS) + f"  {overall:7.2f}")
        for h in HORIZONS:
            rows_out.append(dict(
                run_date=args.date, run_ts=now, method_tag=f"{name}_h{h}_{scope_tag}", status="ok",
                encoder=args.encoder if name != "persistence" else "-",
                head="mlp" if name != "persistence" else "none", fill=args.fill, scope=scope_tag,
                horizon=h, **per_h[h], fit_windows=n_fit or "", eval_rows=n,
                fallback_rows=int(n - len(idx)), ridge_lambda="", runtime_s=runtime,
                notes=(f"init={args.init} cov={not args.no_covariates} best_epoch={best_epoch} "
                       f"val_overall={best_val:.3f}" if name != "persistence"
                       else "reference baseline on identical rows")))
    append_results(out_dir / f"{EXPERIMENT}_{scope_tag}_results.csv", rows_out)
    del He, Ce

    # --------------------------------------------------------- full template
    pred_path = None
    if args.predict_full:
        print("\n[4/4] Predicting the full template")
        rows = template
        if args.only_source:
            keep = {s for s in re.split(r"[,\s]+", args.only_source) if s}
            rows = rows[rows["source_file"].isin(keep)]
            if rows.empty:
                sys.exit(f"--only-source {sorted(keep)} selected no rows")
        rows = rows.reset_index(drop=True)
        nf = len(rows)
        preds = np.full((nf, len(HORIZONS)), np.nan)
        anchor = np.full(nf, np.nan)
        for idx_c, hc, xc in iter_eval_chunks(data_dir / "test.parquet", rows, fill=args.fill,
                                              chunk=args.eval_chunk, verbose=False, extra=cov):
            anchor[idx_c] = hc[:, -1]
            d = predict_deltas(model, hc.astype(np.float32), norm(xc).astype(np.float32), device,
                               args.eval_batch)
            preds[idx_c] = np.clip(hc[:, -1:].astype(np.float64) + d, *GLUCOSE_RANGE)
        gap = ~np.isfinite(anchor)
        preds[gap] = np.nanmedian(anchor) if np.isfinite(anchor).any() else 120.0
        shard = ("_" + re.sub(r"[^A-Za-z0-9]+", "-", args.only_source).strip("-")) if args.only_source else ""
        full_tag = f"livefull{nf}{shard}"
        pred_path = out_dir / "preds" / f"{tag}_{full_tag}.parquet"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        write_predictions(pred_path, rows, preds, comp_dir / "template.parquet")
        append_results(out_dir / f"{EXPERIMENT}_{full_tag}_results.csv", [dict(
            run_date=args.date, run_ts=datetime.now().isoformat(timespec="seconds"),
            method_tag=f"{tag}_h{h}_{full_tag}", status="ok", encoder=args.encoder, head="mlp",
            fill=args.fill, scope=full_tag, horizon=h, n=nf, rmse="", mard="", dts_a="", dts_b="",
            dts_c="", dts_d="", dts_e="", fit_windows=n_fit or "", eval_rows=nf,
            fallback_rows=int(gap.sum()), ridge_lambda="", runtime_s=round(time.time() - t_start, 1),
            notes="predictions written; scored by run.py or server-side") for h in HORIZONS])
        print(f"    {nf:,} rows, {int(gap.sum()):,} without usable history -> {pred_path.relative_to(root)}")
    else:
        print("\n[4/4] Full template skipped (pass --predict-full for a submission parquet)")

    cfg_path = log_dir / f"{tag}_{scope_tag}.json"
    cfg_path.write_text(json.dumps(
        {**vars(args), "tag": tag, "device": str(device), "fit_windows": n_fit,
         "best_epoch": best_epoch, "best_val_overall": best_val, "trainable_params": n_train_p,
         "rmse": summary, "runtime_s": round(time.time() - t_start, 1),
         "peak_rss_gb": round(peak_rss_gb(), 2)}, indent=2, default=str), encoding="utf-8")
    print(f"\n  results    -> {(out_dir / f'{EXPERIMENT}_{scope_tag}_results.csv').relative_to(root)}")
    print(f"  run config -> {cfg_path.relative_to(root)}")
    if pred_path is not None and not args.only_source:
        print(f"  validate   -> python run.py {pred_path.relative_to(root)} --competition live --horizon all")
    print(f"  runtime    -> {time.time() - t_start:.0f}s, peak RSS {peak_rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
