# End-to-end fine-tuning of CGM-JEPA with insulin and carbs

## 1. What changed

- **New `finetune_jepa.py`** — fine-tunes the CGM-JEPA encoder as the CGM branch
  of a multi-horizon forecaster:
  - encoder (published weights / random / frozen) → last-hour token + mean token (192);
  - CGM summaries (last 2 h, Δ5/15/30/60, last-hour SD, 24 h mean; 30);
  - all 188 `Covariates` columns → signed log1p → standardised (train split) → MLP (256);
  - concat → LayerNorm → MLP(512, 512) → 4 changes from G(t), scaled 1/50.
  Loss = MSE over the four horizons with equal weight (= pooled overall RMSE).
  AdamW, head lr 1e-3, encoder lr 1e-4 (`--encoder-lr-mult 0.1`), wd 0.01, 3%
  warmup + cosine, grad clip 1, early stop on participant-disjoint validation
  (10% of participants, patience 2). Device auto: cuda > mps > cpu.
  - Controls: `--init {pretrained,scratch,frozen}`, `--no-covariates`.
  - Outputs: `checkpoints/<date>/jepa_finetune/<tag>.pt` (best val; weights +
    covariate normaliser + args), `logs/<date>/jepa_finetune/<tag>_epochs.csv`,
    `…/<tag>_livesub<n>.json`, results rows on the 200 k subsample (+ persistence)
    in `results/<date>/jepa_finetune/`, and with `--predict-full` a template-complete
    parquet in `…/preds/`. `--checkpoint` re-scores / predicts without training.
  - Tag: `ftjepa_<encoder>_<init>_<cov|nocov>_<fill>[_<suffix>]_h<H>_<scope>`.
  - Training windows cached once as .npy under `cache/finetune_windows/<key>/`
    (key = row groups, per-participant cap, fill, seed); eval subsample likewise.
- **New `covariates.py`** — `Covariates` and `cgm_features` moved out of
  `probe_covariates.py` unchanged, so both drivers share one definition.
  `probe_covariates.py` imports them; verified result-identical (all 11 rows of a
  proxy run match the pre-move run to the digit).
- `.gitignore`: `cache/`, `checkpoints/`.

## 2. Why

Victor: *"yes 20-25 is the right target with some even achieving 15 rmse (idk if
legitimate). lets write up the code to first finetune cgm-jepa with the
competition training data."* Frozen JEPA features stopped helping once insulin
and carbs were present (`richnojepa` gbm2000 27.90 ≤ `rich` 28.08), so the open
question is whether updating the encoder changes that.

## 3. Decisions made

- **Covariates in the model, all future columns for every horizon.** The target is
  the with-future score; the competition supplies the whole window's doses, so a
  30-min output may see doses up to t+120. Differs from the probe (≤ t+h), so
  probe and fine-tune numbers are not a like-for-like feature comparison.
- **Encoder input stays raw mg/dL.** The published weights were trained on it.
- **Encoder lr at 0.1× the head** — protect pretrained features early; the
  `scratch` control uses the same multiplier so the comparison is about init only.
- **Late fusion with the same token pooling as the frozen baseline** (last + mean),
  so frozen-vs-fine-tuned differs only in whether the encoder moves. Multimodal
  input channels into the encoder are a later step.
- **Validation holds out participants**, not windows: 24 h windows on a 15-min
  stride overlap ~97%.
- **Scored on the same 200 k subsample** as `predict_jepa` / `probe_covariates`,
  with the same median fallback for the 209 unwindowed rows, so the overall number
  is directly comparable with 27.90.
- **Windows cached.** Building them from `train.parquet` is the slow part; every
  later run with the same sampling key reuses them.

Verification (container, proxy data = test.parquet as the fit set, **not
results**): all six modes run (pretrained/scratch/frozen × cov/nocov, suffix);
trainable counts 998,720 / 625,088 frozen / 752,960 no-cov; loss decreases and
early stopping fires; `--checkpoint` re-scores to the identical numbers;
`--predict-full --only-source` writes a NaN-free parquet with the template schema.
CPU throughput ~1.8 k windows/s on 2 threads. **MPS is untested** (no Apple GPU
here); `--device cpu` is the fallback.

## 4. What did NOT change / out of scope

- `predict_jepa.py`, the probe's variants and results, `run.py`, `metrics.py`.
- No real-data training run yet (train.parquet cannot reach this container).
- `assemble_submission.py` only reads `jepa_zeroshot` preds; fine-tune shards
  would need it generalised. Unneeded for a whole-template run.
- The median fallback for sparse-cadence rows is kept for comparability.

## 5. Follow-ups

- First run: `pretrained` vs `frozen` vs `scratch`, same data; then the winner
  with `--predict-full` and `run.py`, and submit.
- If fine-tuning wins: larger per-participant cap, longer training, multimodal
  input channels into the encoder, horizon-trajectory output (24 steps).
