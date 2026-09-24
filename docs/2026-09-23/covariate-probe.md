# Covariate probe: what insulin and carbs add to frozen CGM-JEPA

## 1. What changed

- **New `probe_covariates.py`** — scored diagnostic on the live subsample, not a
  submission path. Same fit windows and same template rows as
  `predict_jepa.py --scope sub`; writes
  `results/<date>/jepa_covariates/jepa_covariates_livesub<n>_results.csv` (the
  `RESULT_FIELDS` schema) and `logs/<date>/jepa_covariates/*.json` (with an
  `rmse` summary including pooled "overall"). No prediction parquet.
  Variants × heads:
  - `jepa` (ridge) — must reproduce `predict_jepa` subsample numbers exactly.
  - `jepahist` (ridge) — + insulin/carb history.
  - `jepahistfut` (ridge, gbm) — + known future insulin/carbs up to t+h.
  - `cgmhistfut` (ridge, gbm) — last 2 h raw CGM + same covariates, no JEPA (control).
  - persistence on identical rows.
  Tags: `jepacov_<encoder>_<variant>_<head>_<fill>_h<H>_livesub<n>`.
- **`jepa_windows.py`**: `build_fit_set` and `iter_eval_chunks` take an optional
  `extra` hook (`columns` + `__call__(frame, origins_ns)`), returning/yielding its
  features alongside. Default `None` leaves both unchanged; `_frame_columns`
  refuses a hook that reads CGM.

## 2. Why

Victor: *"the 20-25 top rmse results are average across all window sizes and
includes carb and insulin data. is there a quick way to see what cgm-jepa would
perform like given this extra infromation on insulin and carb?"* Our pooled
overall is 37.35 (best config).

## 3. Decisions made

- **Covariates = windowed sums, not an IOB/COB model.** Per channel (bolus U,
  basal U, carbs g; NaN → 0): history bins (t-30,t], (t-60,t-30], (t-120,t-60],
  (t-240,t-120], (t-360,t-240]; future bins (t,t+30] … (t+90,t+120]; plus each
  channel's observed fraction over 6 h, so "none logged" ≠ "not recorded" (carbs
  are absent for IOBP2/AZT1D/CTR3/ShanghaiT1DM). Bins let the head learn the
  action curve instead of assuming one. `insulin` column skipped: it is basal + bolus.
- **Future bins only up to the horizon.** Doses after t+h cannot move G(t+h), and
  in closed-loop data they partly encode it. Even within the horizon, automated
  basal reacts to post-t CGM, so `jepahistfut` is an upper bound on AID datasets
  (Loop, IOBP2, PEDAP are AID). Allowed by the rules; reported separately.
- **Joint ridge for horizon-independent variants, per-horizon otherwise** — joint
  keeps `jepa` bit-identical to `predict_jepa`; future-bin variants need a
  per-horizon feature mask.
- **GBM = scikit-learn `HistGradientBoostingRegressor`**, not LightGBM: its wheel
  bundles OpenMP on macOS. 500 iters, lr 0.05, 31 leaves, early stopping on a
  random 10% (overlapping windows make that stop a little optimistic). GBM only on
  the two full-covariate variants — the question is the ceiling.
- **Same fallback as `predict_jepa`** (median), so rows are comparable; not the fix.

Verification: the hook matches a brute-force pandas computation on a synthetic
frame with gaps, NaNs and shuffled rows (0/150 mismatches). On proxy data
(test.parquet as fit set — **not a result**), the `jepa` row reproduces
`predict_jepa` 22.87/36.89/45.18/50.43 exactly, and `predict_jepa` predictions
are bit-identical before/after the `jepa_windows.py` change.

## 4. What did NOT change / out of scope

- `predict_jepa.py`, the sweep, the result files, `run.py`, `metrics.py`.
- No real-data run yet: `train.parquet` exceeds the transfer cap, so Victor runs
  it on the Mac.
- Annual set: whether its −1 masking hides future insulin/carbs is unchecked; if
  it does, only `jepahist` transfers.
- Participant overlap still unfiltered.

## 5. Follow-ups

- Run on the Mac; if `gbm` beats `ridge` by a wide margin, the next step is a
  GBM head in the submission path (full template, new tags).
- If `cgmhistfut` ≈ `jepahistfut`, the frozen encoder adds nothing once
  covariates are present — decide before investing in fine-tuning.
