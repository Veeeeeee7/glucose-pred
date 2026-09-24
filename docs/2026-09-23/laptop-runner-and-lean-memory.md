# Laptop runner, and a leaner eval path

## 1. What changed

**New**

- `run_jepa_local.sh` — scheduler-free driver for a laptop. Runs each
  (encoder × fill) **sequentially**: `sub` → `full` → `run.py` validation of the
  full parquet. Required, format-checked `--date`. Preflights: the interpreter
  can import numpy/pandas/pyarrow/torch/safetensors, data present, weights
  present, results tree empty (`ALLOW_EXISTING=1` to override). Re-execs under
  `caffeinate -i` on macOS. Threads default to the performance-core count
  (`hw.perflevel0.physicalcpu`). Defaults: `EVAL_CHUNK=10000`, `BATCH_SIZE=512`,
  `--device cpu`, `EVAL_ROWS=200000`, `FIT_ROW_GROUPS=24`. Logs tee'd to
  `logs/<date>/jepa_zeroshot/<enc>_<fill>_<shard>_local-<HHMMSS>.out`.
  bash 3.2-compatible (macOS system bash); `shellcheck` clean.

**Modified — all result-neutral (verified bit-identical, §3)**

- `jepa_windows.py`
  - `read_template()` (new): reads only `id, source_file, date`, with the two
    string keys as dictionaries → categoricals. 430 MB → 29 MB for the live template.
  - `participant_positions()` (new): replaces the per-row Python `pos_of` dict
    and the two `astype(str)` zips in `iter_eval_chunks` with a groupby on
    categorical codes.
  - `extract_from_grid`: grid lookup done in `GATHER_BLOCK = 8192`-origin blocks,
    bounding its int64 `(N, 288)` temporaries.
  - `TEMPLATE_KEY` constant.
- `predict_jepa.py`
  - `embed_features()`: `features(embed_windows(...))` one forward batch at a
    time, so `(N, 24, 96)` patch embeddings never exist at chunk size. Same batch
    boundaries as before. Used for both fit and eval.
  - `load_truth()`: sub-phase target join moved to Arrow (targets on the probe
    side, requested rows on the build side, `right outer`, re-sorted by position).
  - `write_predictions()`: builds the output in Arrow and casts to the
    template's schema (carries its pandas metadata → file reads back with the
    same dtypes as before).
  - Fit arrays deleted after the ridge solve; `fit_windows` now from `n_fit`.
  - **Log-schema addition:** run-config JSON gains `peak_rss_gb`; stdout gains a
    `peak RSS` line. Results-CSV schema **unchanged**.
- `run_jepa_cluster.sh`: ALLOCATION comment updated to the measured figures. No
  behavior change.
- `CLAUDE.md`, `EXPERIMENTS.md`: synchronized.

## 2. Why

Victor: *"The cluster currently is quite busy. is it possible for you to make
another experiment running path where the memory requirement a lot lower
(loading less data at a time). This will allow me to run it on my macbook? I
have an apple m4 pro chip with 24 gbs of memory."*

First real-data measurement of the old path (full live template, 2.65 M rows):
**3.50 GB peak RSS**, not the ~1.3 GB previously documented (that figure came
from a synthetic-window benchmark). Breakdown of what did not stream:

| Held for the whole run / chunk | Size |
|---|---|
| template as pandas objects, plus a `reset_index` copy | ~430 MB × 2 |
| `pos_of` dict of Python int lists + key tuples | ~200 MB |
| patch embeddings at chunk 50 k, plus `np.concatenate` copy | 461 MB × 2 |
| grid-lookup int64 temporaries at chunk 50 k | ~600 MB transient |
| sub phase only: targets as pandas + pandas merge | ~700 MB (Arrow `left` join also ~700 MB — hashes the big side) |

It already fit a 24 GB laptop; this cut makes the footprint small enough that
the laptop is a comfortable place to run the whole sweep while other work runs.

## 3. Decisions made

- **Make the one path leaner, rather than add a second Python driver.** A
  low-memory fork of `predict_jepa.py` would drift from the cluster path. Every
  change is result-neutral, so the cluster gains the same savings for free; the
  "other path" is the driver script, not the model code.
- **Equivalence verified bit-for-bit, not by metric.** `train.parquet` (1.16 GB)
  exceeds the session's transfer cap, so tests ran with `test.parquet` standing
  in as the fit parquet (`proxdata/`). That changes the head, so these runs are
  **not results** and were not kept — but old vs new code see identical inputs:
  - sub, 19,999 rows: prediction parquets `DataFrame.equals` → True; every metric
    column of the results CSV identical. Persistence reproduced 25.21 / 39.72 /
    48.87 / 55.19 exactly (confirms the same row sample).
  - sub, `--eval-chunk 50000` vs `5000`: identical.
  - full, 2,648,987 rows, `--eval-chunk 50000` vs `10000`: identical; the 10 k
    output passes `run.py --competition live --horizon all` ("Format is valid").
  - full, old code at chunk 50 k vs new code at chunk 10 k, on the seven
    non-Loop/ReplaceBG sources (256,776 rows, including all 2,562 fallback
    rows): identical.
- **`EVAL_CHUNK=10000` on the laptop, 50000 left on the cluster.** It was already
  documented as result-neutral; now tested. 10 k saves ~0.1 GB and costs nothing
  measurable.
- **Sequential configurations on the laptop**, unlike the cluster's one-job-each
  fan-out: each run already uses all performance cores for the encoder.
- **CPU only, no MPS.** The encoder is 522 k parameters; `resolve_device` has no
  MPS path and `check_device_parity` has never met Apple's backend. Not worth a
  new source of cross-machine drift.
- **P-core thread count, not all cores.** E-cores are slower; a pool sized to
  every core stalls on them at each sync point.
- **`caffeinate -i`** so idle sleep can't suspend a multi-minute run. Does not
  prevent lid-close sleep on battery — stated in the script header.
- **Validation runs `run.py` as-is.** Competition-supplied; not modified.

Measured peak RSS (Linux, 2 cores, same inputs):

| Run | Old | New |
|---|---|---|
| sub, 19,999 rows, chunk 50 k | 2.07 GB | 0.95 GB |
| sub, 200,000 rows, chunk 10 k | — | 1.04 GB |
| full, 2,648,987 rows, chunk 50 k | 3.50 GB | 1.47 GB |
| full, 2,648,987 rows, chunk 10 k | — | 1.38 GB |

Remaining floor is ~0.3 GB for torch/pyarrow imports plus the row-scaled arrays
(predictions, anchors, origins, categorical keys: ~150 MB at full size).

## 4. What did NOT change / out of scope

- **No result has changed**, and none of the proxy runs above is a result.
  `results/2026-09-20/` is untouched and still current.
- `run.py`, `metrics.py`, `data/` untouched. `model.py` / `train.py` /
  `prepare_windows.py` untouched.
- Results-CSV schema and method-tag grammar unchanged.
- The macOS path itself has **not** been exercised: the linked Mac's shell is a
  Linux VM with no torch and a full disk. Tested on Linux (bash 5); the script
  avoids bash-4 features but has not met bash 3.2.
- `assemble_submission.py` still reads the template with pandas (~0.5 GB);
  only used after sharded runs.
- **Found, not fixed:** the 2026-09-22 cluster sweep (jobs 564114–564117) failed
  every phase at `import numpy` — the cluster `glucose-pred` env is missing its
  packages. No results row was written because the failure precedes
  `predict_jepa.py`'s own logging.
- Participant-overlap exclusion, annual template, and sentinels: still open.

## 5. Follow-ups

- **First laptop run:** `bash run_jepa_local.sh --date <D>`, or one config first
  with `ENCODERS=x_cgm_jepa FILLS=interp`. Check that the sub phase reproduces
  the 2026-09-20 20 k numbers if run with `EVAL_ROWS=20000 FIT_ROW_GROUPS=8`
  (21.17 / 34.90 / 42.36 / 46.57) — macOS torch may differ from Linux in the last
  float bits, so agreement to the printed 2 d.p. is the bar.
- **Read `peak_rss_gb` from the first macOS run's JSON** and replace the Linux
  figures in `CLAUDE.md` if they differ materially.
- **Fix the cluster env** (`pip install -r requirements.txt -r
  requirements-jepa.txt` inside `glucose-pred`) before resubmitting; consider
  adding the same import preflight to `run_jepa_cluster.sh`.

## 6. Later the same day: first macOS run, runner resume, a fallback defect

**First macOS run** (M4 Pro, 10 threads, `x_cgm_jepa` + `interp`, filed under
`--date 2026-09-25` — Victor's choice of partition; the run was on 2026-09-23).
Fit 48,241 windows, λ = 0.01. RMSE mg/dL:

| | 30 | 60 | 90 | 120 |
|---|---|---|---|---|
| sub 200 k, JEPA | 21.10 | 34.73 | 42.23 | 46.34 |
| sub 200 k, persistence | 25.30 | 40.03 | 49.23 | 55.38 |
| full 2,648,987 (`run.py`) | 21.10 | 34.72 | 42.21 | 46.36 |

Sanity: sub ≈ full to 0.02 (proportional sampling works); the 200 k persistence
row matches the Linux proxy run on the same rows exactly; JEPA is slightly
better than the 2026-09-20 20 k run, consistent with 3.5× the fit windows. λ at
the grid floor is not a concern at n = 48 k (the penalty is already negligible).
Peak RSS 1.70 GB sub / 1.82 GB full. Full runtime **1,525 s** — about one Linux
core's worth of encoder throughput despite 10 threads; the 2-core Linux container
did the same pass in 720 s. Unexplained; not a correctness issue.

**Runner change — per-phase skip replaces the tree-level refusal.**
`run_jepa_local.sh` refused any `results/<date>/jepa_zeroshot/` holding CSVs, so
finishing a sweep started one configuration at a time needed `ALLOW_EXISTING=1`
— and because `ENCODERS × FILLS` is a cross product, that would also have re-run
the finished configuration. Now a phase is skipped when its prediction parquet
exists (sub: `…_<comp>sub<n>.parquet`; full: exact `…full<n>[_shard].parquet`,
with `<n>` the template or shard row count). `ALLOW_EXISTING=1` re-runs. Keyed on
the parquet because `predict_jepa.py` writes it immediately before appending
result rows. Validation still runs on a skipped full phase (3 s). Also: the
interpreter preflight now names the missing packages and the interpreter path.
Tested: dry-run and real shard runs, skip and re-run paths; `shellcheck` clean.

**Defect found, not fixed (changes results — needs a decision).** All 2,562
unwindowed live rows are `ShanghaiT1DM` (281/281; CGM every 15 min → 33% of the
5-min grid < `min_valid_frac` 0.5) or `BrisT1D` (2,281/19,530; 15-min stretches).
CGM(t) is observed on every one, yet they are predicted as the population median
131 mg/dL at all horizons. On those rows: RMSE 74.7 vs 32.9 (last-value
persistence) at 30 min, 74.8 vs 52.4 at 60, 75.2 vs 65.3 at 90, 75.3 vs 74.7 at
120. Overall effect with persistence substituted: ≈ −0.10 mg/dL at 30 min,
−0.04 at 60, ≈ 0 beyond. The persistence reference row shares the fallback, so
the JEPA − persistence gap is unaffected. Options, easiest first:

1. **Fallback = last observed CGM** instead of the median. ~10 lines in
   `predict_jepa.py`; touches only these rows; persistence row becomes true
   persistence.
2. **Cadence-aware eval validity** — relax `min_valid_frac` on the eval side only
   (e.g. 0.3) so 15-min series are interpolated and windowed. Every currently
   windowed row already passes 0.5, so only these rows change. The encoder never
   saw interpolated 15-min input in pretraining; check per-source RMSE.
3. Both: 2 for rows that clear the relaxed bar, 1 for the rest.

Any of them changes predictions under an unchanged method tag, which the tag
rule forbids, and re-running into the same date would overwrite the existing
prediction parquet. So a fix ships with a tag change (or a fresh `--date`), and
all four configurations — including the finished one — are run under it.
