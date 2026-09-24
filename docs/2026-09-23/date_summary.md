# 2026-09-23 summary

Four docs today, in order:

1. `laptop-runner-and-lean-memory.md` — `run_jepa_local.sh`; eval path memory
   3.5 → 1.4 GB (bit-identical); first macOS run; per-phase skip; the
   sparse-cadence fallback defect (open).
2. `covariate-probe.md` — `probe_covariates.py`: what insulin/carbs add to frozen
   JEPA (ridge / GBM). Result: 37.36 → 30.49 overall on the 200 k subsample.
3. `covariate-probe-rich.md` — 5-min dose slots, dose-action features, CGM
   summaries; with 506 k fit windows GBM reaches 27.90 (no JEPA) / 28.08 (JEPA).
4. `finetune-jepa.md` — `finetune_jepa.py` (end-to-end fine-tuning with
   covariates, three init controls) and `covariates.py` (shared hook).

Net: best honest number is 27.90 overall (GBM, no JEPA); frozen JEPA adds nothing
once covariates are present; fine-tuning is the next test.
