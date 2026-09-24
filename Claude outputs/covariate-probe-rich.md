# Covariate probe, round 2: rich features and larger fit sets

## 1. What changed

- `probe_covariates.py`
  - `Covariates` hook gains four blocks **appended after** the original 30
    columns (which are unchanged, so earlier tags keep their meaning): `hist_raw`
    (last hour, 5-min slots), `fut_raw` (next 2 h, 5-min slots), `effect`
    (per channel × 2 time constants × past/future × horizon: expected dose
    action inside (t, t+h] under a first-order exponential response), `tod`
    (sin/cos time of day). 188 columns total. `rich_cols(h)` selects everything a
    horizon-h head may use.
  - `cgm_features()`: last 2 h raw CGM, change over 5/15/30/60 min, SD of the
    last hour, 24 h mean.
  - New variants `rich` (JEPA + CGM summaries + rich covariates) and
    `richnojepa` (control). New flags `--variants` (subset) and `--gbm-iters`
    (non-default value goes into the head label, e.g. `gbm2000`).

## 2. Why

Victor, after round 1 (best: `jepahistfut`/gbm overall 30.49 on the 200 k
subsample, vs 37.36 JEPA-only and 43.97 persistence): *"it seems that the
insulin / carb values brings the rmse down to around 33, which is good but not
enough to explain the gap to 20. lets see if we can make some improvements from
here. since future insulin and carb are available for the competition, dont
worry about any overstating of results. the final prediction accuracy we want is
the one with future insulin and carb"*.

## 3. Decisions made

- **Target is the with-future variants.** Victor's call; the AID-upper-bound
  caveat stays in the notes field but no longer gates anything.
- **Why these features.** Round-1 future covariates were 30-min sums; on AID data
  the next 2 h of automated basal at 5-min resolution tracks the controller's
  view of glucose, and a tree cannot form lagged weighted sums of doses itself,
  so the action features hand it that shape directly. Time constants: insulin
  55/120 min, carbs 30/90 min — two per channel so the head can interpolate
  rather than trust one assumed curve.
- **Appended, not reordered.** Round-1 variants read the same columns in the same
  order, so their tags are semantically unchanged.
- **Fit-set size stays a flag** (`--fit-row-groups`, `--fit-per-group`), not a new
  default: round 1 used 24 row groups × 150; the obvious next test is all 128.

Verification: every new column matches a brute-force pandas computation on a
synthetic frame with gaps, NaNs and shuffled rows (0/860 mismatches); no
`rich_cols(h)` index exceeds t+h. On proxy data (not a result) all six round-1
rows reproduce to the digit.

## 4. What did NOT change / out of scope

- Round-1 variants, `predict_jepa.py`, the submission path, result files.
- No submission-path GBM yet; this is still subsample-only.
- Dataset/algorithm identity is not a feature (five test sources are absent from
  the fit set).

## 5. Follow-ups

- Run round 2 on the Mac, at current and at full fit-set size.
- If `rich`/gbm moves materially, port the head + features into a full-template
  driver with new tags.
- If `richnojepa` ≈ `rich`, the frozen encoder is not pulling its weight.
