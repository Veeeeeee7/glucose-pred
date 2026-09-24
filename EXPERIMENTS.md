# EXPERIMENTS.md

**Last synchronized with code: 2026-09-23 (third revision).** Added `finetune_jepa.py`; moved `Covariates` / `cgm_features` into `covariates.py` (result-identical). Earlier the same day: Added `probe_covariates.py` and the `extra` hook on `build_fit_set` / `iter_eval_chunks` (default off, result-neutral). Earlier the same day: added `run_jepa_local.sh`, a
scheduler-free laptop driver. Cut peak memory of the shared eval path, result-
neutrally (verified bit-identical): `read_template` holds only categorical keys,
`participant_positions` replaces the per-row Python index, `extract_from_grid`
gathers in blocks, `predict_jepa.embed_features` reduces patch embeddings to
features per forward batch, and targets are joined and predictions written in
Arrow. Run-config JSON gains `peak_rss_gb`.

*Previously synchronized: 2026-09-20 (second revision) — Replaced
`jepa_windows.build_eval_set` with the bounded-memory generator
`iter_eval_chunks` and split `extract` into `grid_of` + `extract_from_grid`;
`predict_jepa.py` now predicts chunk by chunk and gained `--only-source` and
`--eval-chunk`. Added `assemble_submission.py` and the cluster scripts. The
refactor is result-neutral: the 20 k subsample reproduces its RMSE and windowed
count exactly.*

*Previously synchronized: 2026-09-20 — added the CGM-JEPA zero-shot-encoder
baseline (`jepa_model.py`, `jepa_windows.py`, `predict_jepa.py`) and the
date-partitioned `results/` + `logs/` layout. Nothing in the pre-existing
`model.py` / `prepare_windows.py` / `train.py` family changed.*

This file is **current-only**. It describes what the code does today. History
lives in `docs/YYYY-MM-DD/`.

---

## 0. Spec of record and divergence register

The **competition's own documents outrank this repo's code**: the challenge page
at <https://metabo-net.org/glucose-prediction-challenge>, the upstream
`README.md`, and `data/README.md`. Where the code and those documents disagree,
the documents win and the disagreement is registered here rather than silently
resolved in either direction.

**Open divergences:**

| # | Spec says | Code does | Assessment |
|---|---|---|---|
| 1 | `data/annual_competition/template.parquet` ships with the repo and defines the annual rows | The file is absent from this working tree, and `.gitignore` excludes `*.parquet` so it is untracked | Local tree defect, not a code bug. Recover before any annual work. |
| 2 | Annual inputs mask prediction targets as `-2` and masked forecast-window context as `-1` | No module recognises either sentinel; `jepa_windows` treats every non-timestamped slot as NaN and `-2`/`-1` would be read as glucose values | Real code gap. Must be closed before the annual set is touched. |

Never resolve a divergence quietly. Name it, then ask.

---

## 1. Data

### Source

MetaboNet, downloaded from <https://metabo-net.org>. Multi-source type-1
diabetes CGM with insulin, carbohydrate, and device metadata. Nothing under
`data/` is version-controlled — `.gitignore` excludes `*.parquet` and no parquet
is tracked — so a clone is not runnable until the data is re-fetched.

| File | Rows | Columns | Role |
|---|---|---|---|
| `data/train.parquet` | — | 37 | fitting data; never scored |
| `data/test.parquet` | 22,069,548 | 37 | model inputs for both leaderboards |
| `data/live_leaderboard/template.parquet` | 2,648,987 | 7 | rows to predict, and their order |
| `data/live_leaderboard/targets.parquet` | 2,648,987 | 7 | ground truth, row-aligned with the template |
| `data/annual_competition/template.parquet` | — | 7 | **missing locally** |

Relevant columns: `source_file` (dataset), `id` (participant), `date`
(timestamp, 5-minute spacing), `CGM` (mg/dL), `insulin`, `basal`, `bolus`,
`carbs`, `meal_label`, and `subject_split_across_traintest`.

### Target and scale

Glucose in **mg/dL** at +30, +60, +90, +120 minutes from each origin `t`. On the
5-minute grid those are future steps 6, 12, 18, 24 — zero-based indices
`[5, 11, 17, 23]`, which is what `prepare_windows.SCORED_FUTURE_INDEX` encodes.

### How the live test set was built (upstream, not by us)

From 22,069,548 rows: drop NaN targets (−3,336,797), apply a 24 h lookback
criterion (−10,786,072: CGM gap > 30 min, insulin 0/NaN run > 120 min, no
`carbs > 0` or `meal_label` in window, insufficient lookback, or a NaN
current-slot CGM), then a 15-minute origin stride (−5,297,692), leaving
**2,648,987** rows over 279 `(source_file, id)` pairs from 9 datasets.
`DCLP3`, `DCLP5`, and `Flair` are excluded for having no carbohydrate data;
`T1D-UOM` for MDI-only participants with unsatisfiable insulin coverage.

Consequence for interpretation: every scored sample has the inputs an AID system
would have. Numbers here are **not comparable** to papers reporting on
unfiltered CGM, which contains far more degenerate windows.

### The single load path

Every experiment built on the JEPA family goes through
`jepa_windows.extract_from_grid` — reached by `extract` for a whole participant
frame on the fit side, and by `iter_eval_chunks` on the eval side — which
enforces:

- History is exactly **288 samples** on a strict 5-minute grid ending at `t`
  inclusive. Lookup is by exact timestamp; an absent slot becomes NaN rather
  than shifting the series.
- The eval side streams through `iter_eval_chunks`, so peak memory is set by
  `--eval-chunk` and not by the number of rows scored. `grid_of` is built once
  per participant, never per chunk. The template is read as categorical keys
  only, and the grid lookup runs in blocks of `GATHER_BLOCK` origins; neither
  can change a value.
- A window is kept only if `CGM(t)` is observed **and** ≥ `min_valid_frac`
  (default 0.5) of the 24 h is observed.
- For fit windows, all four future targets must be present.
- Participant blocks spanning two parquet row groups are carried and emitted
  whole, so no window crosses a discontinuity.
- `fill_history` then produces a NaN-free array by one rule applied to every
  method.

`model.py`'s family uses `prepare_windows.py` instead, which additionally
carries the four non-CGM channels and applies the live-like eligibility filters.

### Exclusions not yet applied

`subject_split_across_traintest` marks participants present in both splits. It
is **not** filtered anywhere. Any head fitted on `train.parquet` may therefore
have seen an evaluation participant. Until this is handled, treat JEPA baseline
numbers as an upper bound.

---

## 2. Experiments

### 2.1 `jepa_zeroshot` — frozen CGM-JEPA encoder + ridge readout

**What it measures, and why it exists.** Whether the CGM-JEPA representation
carries information about the *next two hours* of glucose beyond what the current
value already tells you. It exists as a cheap, reliable floor to measure the
team model against, and as a direct test of a published CGM foundation encoder on
a forecasting task it was not built for.

**Why a head is required at all.** CGM-JEPA's published checkpoints
(`CRUISEResearchGroup/CGM-JEPA`, MIT) contain **only an encoder**. Pretraining
minimises an L1 loss between predicted and EMA-target patch embeddings in 96-d
latent space; the JEPA predictor was never released, and no decoder from latent
space to mg/dL exists in the architecture at all. A literal zero-training
forecast from these weights is not possible. The encoder is frozen — no gradient
step touches it — and the readout is the smallest thing that closes the gap.

**Splits.** The fit set comes from `train.parquet` and the eval set from
`test.parquet` restricted to template rows; the two parquets are the
competition's own split. Inside the fit set, the ridge penalty is chosen on a
**participant-disjoint** holdout (20% of participants). This matters: 24 h
windows from one participant on a 15-minute stride overlap by ~97%, so a random
split would tune the penalty against near-copies of the fitting rows.

What is scored: the four horizons, on exactly the sampled template rows. What
stays fixed across rows in a results file: the windowing, the fill rule, and the
row set — so a JEPA row and a persistence row in the same file differ only by
method.

**Model roster.**

| Name | Training scheme |
|---|---|
| `x_cgm_jepa` (default) | frozen; pretrained by the authors with the cross-view Glucodensity objective |
| `cgm_jepa` | frozen; pretrained with the temporal objective alone |
| `persistence` | none; `G(t+h) = G(t)`. Logged on identical rows every run. |

**Architecture as published** (`config.json`, both checkpoints identical):
input (B, 24, 12) raw mg/dL → per-patch `Conv1d(1, 96, k=3, s=3)` → flatten 384
→ `Linear(384, 96)` → sinusoidal positional embedding → 3 pre-norm Transformer
blocks (96-d, 6 heads, MLP ratio 4) → `LayerNorm` → (B, 24, 96). 522,160
parameters. The port in `jepa_model.py` is verified to reproduce the authors'
`models/encoder.py` exactly (max absolute token difference 0.0 on both
checkpoints).

**Readout.** Features are the last-patch embedding (most recent hour) and the
mean-pooled day embedding, 192 dims, standardized. Target is the **change** from
the current value, `Δ_h = G(t+h) − G(t)`, so persistence is the zero solution and
a head that learns nothing degrades to persistence rather than to garbage.
Predictions are `G(t) + W φ(x)`, clipped to the CGM reporting range.

**Hyperparameters.** Every non-default value has a reason; "it is what the
authors pretrained with" is a complete reason.

| Parameter | Value | Reason |
|---|---|---|
| patch size | 12 | Authors' `patch_size`; 12 × 5 min = 1 hour. |
| patches per window | 24 | Authors' day-level window; 288 samples = 24 h. |
| input scaling | raw mg/dL | Pretraining ran with `normalize_x: False`. Z-scoring would move inputs off the distribution the input convolution was fitted on. |
| time features | off | Pretraining ran with `use_time_feature: False`; the embedding exists in the checkpoint but was never trained. |
| `--fill` | `interp` | The encoder saw raw mg/dL. A `-1` sentinel inside a 100–300 mg/dL series is a large outlier to a raw-value convolution, so interior gaps are linearly interpolated and edges clamped. `sentinel` reproduces the authors' padding convention instead. |
| `min_valid_frac` | 0.5 | A window more than half unobserved is interpolation, not measurement. |
| ridge λ | grid `{1e-2 … 1e4}` | Selected on the participant-disjoint holdout, then refit on the full fit set at the chosen value. |
| `--fit-row-groups` | 8 | `train.parquet` has 128 row groups spanning many participants each; 8 spread evenly give cross-dataset coverage without a full 1.1 GB scan. |
| `--fit-per-group` | 150 | Caps any one participant's influence on the head. |
| `--eval-rows` | 20000 | Laptop-sized. The cluster driver raises it to 200000. |
| `--eval-chunk` | 50000 | Windows per streamed chunk. Sets peak memory (full live template: ~1.5 GB at 50 k, ~1.4 GB at 10 k, measured), changes no result. The laptop driver uses 10 k. |
| `--sample-mode` | `proportional` | Preserves the live set's source mix (77% Loop), so a subsample RMSE is comparable to the full-set number. |
| `--seed` | 2026 | Matches `prepare_windows.py` / `train.py`. |

**Metrics and their ceiling effects.** RMSE, MARD, and DTS zones from
`metrics.py`, unmodified. Two cautions: at 30 minutes persistence is already
strong, so the headroom above it is small and RMSE differences there are easy to
over-read; and DTS zone A saturates at short horizons, which makes A% a poor
discriminator there. Compare against the persistence row in the same file, never
against an absolute threshold.

**Outputs.**

```
results/<date>/jepa_zeroshot/jepa_zeroshot_<scope>_results.csv
results/<date>/jepa_zeroshot/preds/<tag_base>_<scope>.parquet
logs/<date>/jepa_zeroshot/<tag_base>_<scope>.json
```

One results row per (method, horizon); append-only, `flock`-guarded.
Tag grammar: `jepa_<encoder>_ridge_<fill>_h<H>_<competition><scope><n>`, with the
reference logging as `persistence_h<H>_<scope>`. `<scope>` is `sub` or `full`.

**Toggles that narrow a run.** `--scope sub` scores a subsample and skips writing
a template-complete file; `--eval-rows`, `--fit-row-groups`, and
`--fit-per-group` set the size. `--scope full` writes every template row, which
is the only output `run.py` will accept. `--only-source` makes a run a SHARD:
its tag and filename carry the shard name, it writes a PARTIAL parquet, and
`assemble_submission.py` must merge the shards onto the template before `run.py`
will accept the result.

**On the cluster.** `run_jepa_cluster.sh` runs one configuration (`sub` then
`full`) under a required `--date`; `submit_jepa_all.sh` fans out one job per
(encoder x fill), with `SHARD_BY_SOURCE=1` splitting the full phase further. The
subsample phase runs in exactly one shard, because it is a whole-mix metric and
a per-source slice of it is comparable to nothing. Allocation: 8 CPUs / 16G / 4h,
CPU-only. See CLAUDE.md section 3 for the site configuration.

**On a laptop.** `run_jepa_local.sh` runs every (encoder x fill) sequentially in
one shell — `sub`, then `full`, then `run.py` on the full parquet — with no
scheduler. `EVAL_CHUNK=10000`, `--device cpu`, threads = performance cores;
`ENCODERS` / `FILLS` narrow it. Peak RSS ~1.4 GB per full pass.

**Reproduction:**

```bash
pip install -r requirements.txt -r requirements-jepa.txt
huggingface-cli download CRUISEResearchGroup/CGM-JEPA \
    --local-dir weights/cgm_jepa_hf --include 'cgm_jepa/*' 'x_cgm_jepa/*'

python predict_jepa.py --date $(date +%F) --scope sub  --eval-rows 20000   # laptop
python predict_jepa.py --date $(date +%F) --scope full                     # submittable
bash run_jepa_local.sh --date $(date +%F)                                  # whole sweep, laptop
python run.py results/<date>/jepa_zeroshot/preds/<file>.parquet \
    --competition live --horizon all
```

### 2.1b `jepa_covariates` — what insulin and carbs add (diagnostic)

`probe_covariates.py --date <D> [--gbm]`, live subsample only, same fit windows
and template rows as `predict_jepa.py --scope sub`. Head inputs per variant:
`jepa` (192 JEPA features; reproduces `predict_jepa`), `jepahist` (+ history
covariates), `jepahistfut` (+ future covariates ending by t+h), `cgmhistfut`
(last 2 h raw CGM + covariates, no JEPA). Covariates: per-channel (bolus, basal,
carbs) sums in history bins to 6 h and future 30-min bins to 2 h, plus 6 h
observed fraction; built by a `Covariates` hook passed to `build_fit_set` /
`iter_eval_chunks` as `extra`, which may not read CGM. Heads: ridge; with `--gbm`,
`HistGradientBoostingRegressor` per horizon on the two future variants. Output
`results/<date>/jepa_covariates/`, tags
`jepacov_<encoder>_<variant>_<head>_<fill>_h<H>_livesub<n>`; the JSON log adds a
pooled `overall` RMSE. Future covariates on AID datasets are an upper bound
(the controller reacts to post-t glucose).

### 2.1c `jepa_finetune` — end-to-end fine-tuning with covariates

`finetune_jepa.py --date <D> [--init pretrained|scratch|frozen] [--no-covariates]`.
Model: CGM-JEPA encoder on raw mg/dL → last + mean token (192) ‖ CGM summaries
(30) ‖ MLP(256) over all 188 `covariates.Covariates` columns (signed log1p,
standardised on train) → LayerNorm → MLP(512, 512) → 4 changes from G(t) / 50.
Every horizon sees future doses to t+120. Loss: MSE, equal horizon weights.
Defaults: 128 row groups × ≤300 windows/participant, 10% participants held out,
AdamW lr 1e-3 (encoder ×0.1), wd 0.01, batch 512, ≤8 epochs, warmup 3% + cosine,
patience 2. Windows cached in `cache/finetune_windows/`. Scored on the standard
200 k subsample with persistence; `--predict-full` writes
`results/<date>/jepa_finetune/preds/<tag>_livefull<n>.parquet`; `--checkpoint`
re-scores without training. Tags
`ftjepa_<encoder>_<init>_<cov|nocov>_<fill>[_<suffix>]_h<H>_<scope>`.

### 2.2 `dual_stream` — causal Gaussian dual-stream forecaster

Pre-existing team model, unchanged by this revision and **not yet run** in this
working tree: there is no `checkpoints/` directory and no `model_data_v1/` shard
tree. It is the only path in this repo to a model that uses insulin and carbs,
which is what the live test set was filtered to reward.

`prepare_windows.py` streams `train.parquet` into NPZ shards of 288×5 history
(CGM, insulin, basal, bolus, carbs), 24×4 future covariates, and 24 CGM targets,
hashing whole `(source_file, id)` groups into train or val so overlapping windows
from one participant never straddle the split. `train.py` fits
`model.DualStreamForecaster` with a masked MSE weighted 2× on the four scored
horizons, and checkpoints on best validation overall RMSE.

It writes to `checkpoints/gluco_v1.pt` and prints to stdout — it does **not**
use the date-partitioned `results/` + `logs/` layout in §3. That gap is
deliberate and unmigrated: existing result files are never moved, and this family
has no result files yet, so the migration is free whenever someone wants it.

---

## 3. Shared plumbing

| Module | Guarantees identical across every method that uses it |
|---|---|
| `metrics.py` | RMSE, MARD, DTS zone assignment. **Competition-supplied; never modified.** Identical to the server's scoring. |
| `jepa_windows.py` | The 288-sample grid, the validity rule, the participant-block carry, and the fill rule — applied the same way to fit and eval sets. |
| `jepa_model.py` | One encoder construction and weight-load path, verified against the authors' implementation, shared by both checkpoints. |
| `predict_jepa.py` | Persistence is computed from the **same** `G(t)` the JEPA head anchors on, over the **same** rows, in the **same** run. |

That last one is what lets a reader trust the JEPA-vs-persistence comparison:
the two rows cannot differ by row set, windowing, or fill.

---

## 4. Scheduling and environment

**Protocol constants** (identical everywhere): patch size 12, 24 patches,
288-sample history, horizons 30/60/90/120, seed 2026, raw mg/dL inputs, no time
features.

**Site-specific knobs** (change these for another machine):
`--batch-size` (encoder forward pass), `--fit-row-groups` and `--eval-rows`
(work per run), `DATA_DIR`, and `--weights`.

The whole suite runs in one environment: `requirements.txt` (competition:
pandas, numpy, pyarrow) plus `requirements-jepa.txt` (torch, safetensors,
huggingface_hub). No driver switches environments mid-run. Inference is CPU-only
and single-process; there is no GPU dependency.

**Cluster note.** Weight download is network-gated. Sync `weights/` with the
code rather than relying on the hub being reachable from compute nodes, and keep
code sync separate from data sync — `data/` is ~1.5 GB and changes rarely. An
rsync `--exclude` list does not read `.gitignore`: `.pylibs/`, `.venv-linux/`,
and `.claude-tmp/` are local scaffolding and must be excluded explicitly.
