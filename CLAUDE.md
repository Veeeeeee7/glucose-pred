# CLAUDE.md

This file is read automatically at the start of every session. It carries the
repo's working conventions and its **current** state — never its history.

Three policies govern everything here:

1. **History lives in `docs/YYYY-MM-DD/`**, never in this file.
2. **Deprecated code is deleted**, not commented out.
3. **Result files are never edited, moved, or deleted.** They are the record.

There is a fourth rule specific to this repo, and it outranks the others:

> **`run.py`, `metrics.py`, and everything under `data/live_leaderboard/` and
> `data/annual_competition/` are competition-supplied. Do not modify them.**
> They define the submission contract. New work goes in new modules that import
> them. If a change to one of these files ever looks necessary, stop and ask.

---

## 1. Documentation policy

**After any major edit, write a new dated doc to `docs/` before ending the
session** — or before moving to unrelated work within a session.

**"Major" means:** adding, removing, or renaming a script or experiment;
changing a results/data schema or file layout; changing a split, CV, or scoring
protocol; changing a method's default behavior; or any refactor touching more
than one file.

**"Major" does not mean:** a one-line bugfix, a comment tweak, a config value,
or exploratory work that didn't land. When in doubt, write it — a short doc
costs almost nothing and a missing one costs a session of re-derivation.

**Naming:** `docs/YYYY-MM-DD/<kebab-case-slug>.md`. The date is a *folder*, so
`ls docs/` sorts chronologically and each day's work is grouped. The slug
carries no date prefix. Multiple docs the same day coexist; add a `-2` suffix
only on a true slug collision.

**The day boundary is 04:00 local, not midnight.** Work at 02:00 on 6/2 files
under `docs/2026-06-01/`. From 04:00, use the current calendar date. Always run
`date` and apply the cutoff rather than inferring today from context — context
is frequently stale.

**Doc structure, five sections:**

1. **What changed** — concrete: files, scripts, schemas touched.
2. **Why** — the motivating problem; quote the ask if it came from Victor.
3. **Decisions made** — wherever you picked between two reasonable options,
   record the choice *and the reason*. Nobody rederives a judgment call from
   the diff.
4. **What did NOT change / out of scope** — so the next session doesn't read
   silence as "handled".
5. **Follow-ups** — flagged but deliberately deferred.

Keep it terse. Dense bullets are preferred — this is a changelog for a future
agent, not a paper. On a day producing three or more docs, add
`docs/YYYY-MM-DD/date_summary.md`.

**The comment/doc split.** The dated doc is the only place history belongs.
Code comments say what the code does and why it is built that way. They carry
no dates, no `docs/` links, no finding numbers, no `CLAUDE.md` cross-references,
and no account of what the code used to be. When a comment guards a real failure
mode, state the failure mode — not the run on which it was discovered. Test: a
reader of a single function should never have to reconstruct project history to
understand it.

## 2. Deprecation policy

- **Deprecated functions and code paths inside active files → delete them.** Git
  history plus the dated doc are the record; a commented-out block is not. Write
  the doc first, naming what was removed and from where, then delete in the same
  session. Leave no dead executable paths.
- **Deprecated driver scripts → delete them too.** No `deprecated/` archive.
- **`CLAUDE.md` carries no deprecated information.** It describes only what is
  active. What a dead thing was, and why it died, lives in the dated doc written
  at deprecation time. Remove its mentions here in the same session — otherwise
  this file quietly becomes fiction, and a file that is sometimes fiction gets
  read as never trustworthy.
- **Result files are never deprecated, moved, or edited.** When an old tree
  holds rows from a retired method, filter by `method_tag` at analysis time.

## 3. Current suite cheat sheet

### The competition

**MetaboNet Glucose Prediction Challenge** (<https://metabo-net.org/glucose-prediction-challenge>).
Forecast glucose 30 / 60 / 90 / 120 minutes ahead. Organised by a consortium
including Stanford, Harvard, Pavia, UCSB, Diabetes Technology Society, JAEB, and
Replica Health. Conditional ("what-if") models that consume known future inputs
are explicitly allowed.

Two formats share one submission contract:

| Format | `--competition` | Template | Targets | `run.py` does |
|---|---|---|---|---|
| Live leaderboard | `live` | `data/live_leaderboard/template.parquet` | shipped | validate **and** score |
| Annual competition | `annual` | `data/annual_competition/template.parquet` | secret, server-side | validate format only |

**Submission contract**, enforced by `run.py`:

- Parquet, with columns `id, source_file, date, pred_30, pred_60, pred_90, pred_120`.
- Exactly the template's rows, in the template's order. No additions, no removals.
- At least one `pred_*` column fully populated. A populated column must contain
  **no NaN**; a skipped horizon must be **entirely** NaN. Partial columns are rejected.

**Metrics** (`metrics.py`): RMSE (mg/dL), MARD (%, zero references excluded),
and the DTS error grid as percentages in zones A–E. `run.py --horizon` defaults
to `60`; `all` also reports a pooled "overall" row over every populated horizon.

**Ranking rule (annual).** Two champions: one ranked by **MARD** (RMSE breaks
ties), one by **DTS error-grid zone**. Secondary, Tertiary and submission
time break ties. RMSE is only a tiebreaker; every head here is trained on MSE. Final annual rankings use the
**average of the per-horizon metric over 30/60/90/120 min** — not the pooled
"overall" row `run.py` prints. The live board displays 60-min only. Best
reported average is ~20 RMSE (per Victor).

**Annual competition key dates.** Submissions opened 2026-08-06. **Deadline
2026-10-05 23:59 AoE.** Results notified 2026-10-09, presented at the Diabetes
Technology Meeting 2026-10-29, leaderboard published 2026-10-31. Submit at
<https://metabonetglucose-leaderboard.hf.space/>.

### Repo lineage

Forked from `replicahealth/glucose-prediction-submission-toolkit`.
`origin` = `Veeeeeee7/glucose-pred`, `upstream` = `OwenTucker/glucose-pred`.
`README.md` is the upstream toolkit's and documents the submission contract for
an outside reader; treat it as inherited documentation and keep it in step with
any protocol change rather than letting it drift.

### What runs today

| Module | Kind | Role |
|---|---|---|
| `run.py` | **competition-supplied** | validate, and for `live` score, a submission |
| `metrics.py` | **competition-supplied** | RMSE, MARD, DTS error grid |
| `model.py` | team model | `DualStreamForecaster` — causal Gaussian dual-stream forecaster |
| `prepare_windows.py` | team model | streams `train.parquet` into NPZ shards for `train.py` |
| `train.py` | team model | trains `DualStreamForecaster` on those shards |
| `jepa_model.py` | JEPA baseline | CGM-JEPA encoder, ported so published checkpoints load |
| `jepa_windows.py` | JEPA baseline | parquet → 24 h history / future CGM pairs |
| `predict_jepa.py` | JEPA baseline **driver** | frozen encoder + ridge head → predictions + results |
| `covariates.py` | shared | insulin/carb covariate hook (`Covariates`, 188 cols) and CGM summaries; never reads CGM past t |
| `finetune_jepa.py` | JEPA fine-tune **driver** | encoder (pretrained / scratch / frozen) + covariate MLP → 4 horizons; subsample score, optional full-template parquet |
| `probe_covariates.py` | JEPA diagnostic **driver** | subsample only: JEPA features ± insulin/carb covariates, ridge and GBM heads, vs a no-JEPA control |
| `assemble_submission.py` | JEPA baseline | merges per-shard parquets onto the template + `PROVENANCE.md` |
| `run_jepa_local.sh` | laptop **driver** | every (encoder × fill) in sequence: `sub`, `full`, `run.py`; required `--date`, no scheduler |
| `run_jepa_cluster.sh` | cluster **driver** | one configuration, `sub` then `full`, required `--date` |
| `submit_jepa_all.sh` | cluster fan-out | one job per (encoder × fill); `SHARD_BY_SOURCE=1` splits further |
| `sbatch_overrides.sh` | cluster | sourced; env → sbatch CLI flags that beat the driver header |
| `upload_to_cluster.sh` | cluster sync | allow-list code push (+ `weights/`) |
| `upload_data_to_cluster.sh` | cluster sync | `data/` push, guarded on template/targets row counts |
| `download_from_cluster.sh` | cluster sync | one date's `results/` + `logs/` down |

The two experiment families are independent: they share `metrics.py` and the
data layout and nothing else. `model.py`/`train.py` consume five history
channels (CGM, insulin, basal, bolus, carbs); the JEPA baseline is CGM-only,
because the published encoder was pretrained on CGM alone.

`predict_jepa.py` takes a **required `--date YYYY-MM-DD`**, never defaulted, and
reads `DATA_DIR` (default `data`). It fail-fast-validates its inputs before
doing any work. Key toggles: `--scope {sub,full}`, `--encoder
{x_cgm_jepa,cgm_jepa}`, `--eval-rows`, `--sample-mode {proportional,balanced}`,
`--fit-row-groups`, `--fit-per-group`, `--fill {interp,sentinel}`.

### Cluster

Site configuration is environment-only: `GP_REMOTE_USER` / `GP_REMOTE_HOST` /
`GP_REMOTE_DIR` (defaults `vmli3` / `cirrostratus.it.emory.edu` /
`/users/vmli3/glucose-pred`), `GP_REMOTE_DATA_DIR` (default
`/scratch/vmli3/glucose-pred/data`), `GP_ENV` for the conda environment
(default `glucose-pred`), `DATA_DIR` for the data root.

**Data lives on scratch, code lives in home.** `upload_data_to_cluster.sh`
pushes the 1.5 GB of parquet to `GP_REMOTE_DATA_DIR` and symlinks
`<code tree>/data` at it, so every driver keeps working with the plain
`DATA_DIR=data` default and nothing has to know where the bytes are. Scratch is
typically purged on a schedule: when a fail-fast data guard trips on a tree that
ran last month, suspect the sweep before the code and re-run the upload.

Code goes up, results and logs come down; never a two-way sync over a results
tree that running jobs are appending to. The code push is an **allow-list** —
root-level `*.py`, `*.sh`, `*.md`, `requirements*.txt`, plus `weights/`. A
block-list fails open, and `data/` is 1.5 GB of clinical CGM that rsync would
push despite `.gitignore`, which rsync never reads. `weights/` is the one
directory that ships (8 MB, and compute nodes often cannot reach the Hugging
Face hub).

The default allocation is **CPU-only**: 8 cores / 16G / 4h on `oxf-c64-m512`
under `--account=general`. Measured, not guessed — the encoder sustains ~2,100
windows/s/core and peak RSS on the full live template is ~1.5 GB at the default
`--eval-chunk` of 50,000 (real data, Linux), because the eval path streams.
`--eval-chunk` is the memory knob and changes no result (tested bit-for-bit). Never shorten `WALLTIME` below a job's real runtime: job size carries no
priority weight on this cluster, so a long walltime costs nothing.

**A GPU is never required and is often still the right choice.** The encoder is
522 k parameters, so the card is idle most of the run — but queue time dominates
the wall clock, and an idle GPU node beats a full CPU partition regardless. Ask
for one explicitly (the driver header requests none):

```
GPUS=1 PARTITION=rp6b-1-gm96-c8-m64 MEM=48G CPUS=8 bash submit_jepa_all.sh --date <D>
```

`--device auto` finds the card; `MEM` stays under that node's 64 GB. Windowing,
gap-filling and the ridge solve are all host-side and are the larger half of the
full-template wall clock, so the CPU count still matters on a GPU node. TF32 is
disabled explicitly, because it would otherwise make the same window embed
differently on a GPU node than on a CPU node — and `check_device_parity` asserts
the card reproduces the CPU forward pass at startup, so an accelerator can never
silently change a result.

`ACCOUNT` must match `PARTITION` or submission is rejected outright; check with
`sacctmgr -nP show assoc user=$USER format=account,partition`.

The full run is four jobs, one per (encoder × fill). `SHARD_BY_SOURCE=1` splits
each configuration's full phase one job per source dataset, in which case the
shards must be merged by `assemble_submission.py` before `run.py` will accept
them — a shard is a partial submission and its tag says so.

### Laptop

`run_jepa_local.sh --date <D>` is the scheduler-free path, for when the cluster
queue is full. It runs each (encoder × fill) **sequentially** — `sub`, `full`,
then `run.py` on the full parquet — because each run already uses every
performance core. Defaults differ from the cluster where a laptop differs:
`EVAL_CHUNK=10000`, `--device cpu`, threads = performance-core count. It
preflights the interpreter's imports first, re-execs under `caffeinate -i` on
macOS (lid-close sleep on battery still stops it), and skips any phase whose
prediction parquet already exists (`ALLOW_EXISTING=1` re-runs it) — so a sweep
started one configuration at a time is finished by re-running the plain command.
On an M4 Pro, `THREADS=4` runs a full pass in ~10 min versus ~25 min at the
default 10 threads (oversubscription on small matmuls) — pass it explicitly. Peak
RSS ~1.9 GB; every run logs its own as `peak_rss_gb` in the
run-config JSON. Activate the env first or pass `PYTHON=`.

### Shared plumbing guarantees

`jepa_windows.extract_from_grid` is the single path from parquet to a model
input, used for both the fit set and the eval set. It guarantees, identically
for every method built on it:

- History is exactly 288 samples on a strict 5-minute grid ending at `t`
  inclusive; slots with no matching timestamp become NaN rather than being
  silently shifted.
- A window is kept only if the sample at `t` exists and ≥ `min_valid_frac` of
  the 24 h is observed.
- Participant blocks split across parquet row groups are carried and emitted
  whole, so no window is ever built across a discontinuity.
- `fill_history` produces a NaN-free array by one declared rule for all methods.
- The eval path **streams** (`iter_eval_chunks`): peak memory is set by
  `--eval-chunk`, not by how many rows are scored, so a 20 k laptop subsample
  and the 2.65 M-row full pass run the same code at the same footprint. The
  template is held as categorical keys only (`read_template`), patch embeddings
  are reduced to features one forward batch at a time, and targets are joined
  in Arrow — so nothing row-scaled is held as Python strings.

Because both sets go through it, a row-to-row comparison in a results file
reflects the method, not the windowing.

### Data layout

Inputs are flat and unpartitioned under `data/`; outputs are date-partitioned.

```
data/train.parquet                        1.1 GB   MetaboNet train split
data/test.parquet                         262 MB   22,069,548 rows x 37 cols, model inputs
data/live_leaderboard/template.parquet    2,648,987 rows
data/live_leaderboard/targets.parquet     2,648,987 rows, row-aligned with the template
data/annual_competition/template.parquet  MISSING locally — see §4
data/persistence.parquet                  34 MB    persistence-baseline submission
weights/cgm_jepa_hf/{cgm_jepa,x_cgm_jepa}/ CGM-JEPA checkpoints from Hugging Face

results/<date>/<experiment>/<experiment>_<slice>_results.csv
results/<date>/<experiment>/preds/*.parquet
logs/<date>/<experiment>/*.json
```

Result rows are **append-only** and `flock`-guarded on a `.<name>.csv.lock`
sidecar; a re-run adds rows rather than replacing them, so aggregate by
`method_tag` and take the latest. Every row carries a `method_tag` and a
`status` field (`ok` / `FAILED`) — a logged failure beats a missing row,
because a gap is indistinguishable from a job that never started.

Tag grammar: `jepa_<encoder>_<head>_<fill>_h<H>_<competition><scope><n>`, e.g.
`jepa_x_cgm_jepa_ridge_interp_h60_livesub20000`. The persistence reference logs
as `persistence_h<H>_<scope>`. Never reuse a tag whose semantics changed.

### Known data hazards

- **`*.parquet` is gitignored and nothing under `data/` is tracked.** A fresh
  clone has no data. Re-fetch with `git lfs pull` for the templates/targets and
  from metabo-net.org for train/test. Do not assume a clone is runnable.
- **Participant overlap between splits.** `test.parquet` carries a
  `subject_split_across_traintest` boolean. Any head fitted on `train.parquet`
  and evaluated on `test.parquet` can therefore see the same participant on both
  sides. This is not yet excluded anywhere — see §4.
- **The live test set is 77% `Loop`.** Of 2,648,987 rows: Loop 2,035,382,
  ReplaceBG 356,829, IOBP2 146,525, PEDAP 63,213, BrisT1D 19,530, CTR3 10,520,
  HUPA-UCM 8,957, AZT1D 7,750, ShanghaiT1DM 281. A row-weighted metric is
  mostly a Loop metric. `--sample-mode proportional` preserves that mix so a
  subsample RMSE is comparable to the full run; `--sample-mode balanced` breaks
  it deliberately, for per-source diagnostics only.
- **The test set is pre-filtered for multi-modal coverage**: 24 h lookback, CGM
  gap ≤ 30 min, insulin gap ≤ 2 h, ≥ 1 carb/meal event, 15-minute origin stride.
  `DCLP3`, `DCLP5`, `Flair`, and `T1D-UOM` are excluded entirely by those rules.
  Absolute numbers here are **not** comparable to papers reporting on unfiltered
  CGM.
- **CGM-JEPA licence and provenance.** Encoder weights come from
  `CRUISEResearchGroup/CGM-JEPA` (MIT). The download is network-gated, so a
  cluster run needs `weights/` synced or the hub reachable.

## 4. Project state — as of 2026-09-23

**Current result trees:** `results/2026-09-20/jepa_zeroshot/` (20 k subsample)
and `results/2026-09-25/jepa_zeroshot/` (laptop sweep, all four configurations,
full template, partition dated ahead of the run).
Nothing is superseded and nothing is known-invalid.

**First full-template result** — `x_cgm_jepa` + ridge, `--fill interp`, 48,241 fit
windows (24 row groups), all 2,648,987 live rows, scored by `run.py`: RMSE
21.10 / 34.72 / 42.21 / 46.36 mg/dL at 30 / 60 / 90 / 120 min, overall 37.35. The
200 k proportional subsample from the same run gives 21.10 / 34.73 / 42.23 /
46.34 against persistence 25.30 / 40.03 / 49.23 / 55.38. Format-valid; not yet
submitted.

Full-template `run.py` RMSE for the whole sweep (30 / 60 / 90 / 120, overall):

| Config | 30 | 60 | 90 | 120 | Overall |
|---|---|---|---|---|---|
| `x_cgm_jepa` + `interp` | 21.10 | 34.72 | 42.21 | 46.36 | 37.35 |
| `cgm_jepa` + `interp` | 21.09 | 34.72 | 42.22 | 46.37 | 37.36 |
| `x_cgm_jepa` + `sentinel` | 22.35 | 35.69 | 43.01 | 47.18 | 38.24 |
| `cgm_jepa` + `sentinel` | 22.44 | 35.72 | 42.99 | 47.12 | 38.24 |

The two encoders are indistinguishable under a linear readout; `interp` beats
`sentinel` by ~1 mg/dL at every horizon.

**First real-data result** — `--scope sub`, 19,999 live template rows sampled
proportionally, `x_cgm_jepa` + ridge, `--fill interp`, 13,761 fit windows, 19.2 s
on a laptop. RMSE mg/dL, JEPA vs persistence on identical rows:

| Horizon | JEPA+ridge | Persistence | Δ |
|---|---|---|---|
| 30 min | 21.17 | 25.21 | −4.04 |
| 60 min | 34.90 | 39.72 | −4.82 |
| 90 min | 42.36 | 48.87 | −6.51 |
| 120 min | 46.57 | 55.19 | −8.62 |

The frozen representation beats persistence at every horizon, and the margin
grows with horizon — which is the expected shape if the embedding carries real
trajectory information rather than just re-deriving the current level. Read
these as an upper bound until the participant-overlap hazard below is handled.

**What has never been run:**

- **Live leaderboard: one submission** (2026-09-24, "TAIL Lab"):
  `results/2026-09-25/jepa_finetune/preds/ftjepa_x_cgm_jepa_pretrained_cov_interp_livefull2648987.parquet`.
  Board shows 60-min only: RMSE 25.1, MARD 12.9%, DTS A 80.5% — matches local
  `run.py --horizon 60` exactly (25.06 / 12.92 / 80.5). Per-horizon average
  (the ranking metric): RMSE 26.54, MARD 13.79%, DTS A 78.4%.
  The leaderboard ranks on the 60-min horizon, not the pooled overall.
  Nothing submitted to the annual competition.
- **No job has completed on the cluster.** The 2026-09-22 sweep (jobs
  564114–564117) reached Slurm and failed every phase at `import numpy`: the
  cluster `glucose-pred` env lacks the packages. The partition/account pair
  (`oxf-c64-m512` / `general`) is still inherited from CCQ-Dataset-Experiments.
- `train.py` has **never been run** in this working tree: there is no
  `checkpoints/` directory and no `model_data_v1/` shard tree. The
  `DualStreamForecaster` in `model.py` is untrained code, not a trained model.
- The annual competition has never been attempted, and cannot be from this tree
  as it stands.

**Blocking gaps:**

- **`data/annual_competition/template.parquet` is missing.** It is not tracked
  (`.gitignore` excludes `*.parquet`), so `git lfs pull` will not restore it;
  recover it with `git checkout upstream/main -- data/annual_competition/` or by
  re-cloning the upstream toolkit. Without it, `run.py --competition annual`
  exits before validating anything.
- **The annual competition input dataset has not been downloaded.** It is behind
  a sign-in on the challenge page, and its masking convention differs from the
  live set: values to predict are `-2`, masked forecast-window context is `-1`.
  No code here handles those sentinels yet.
- **Participant overlap is not excluded.** The ridge head in `predict_jepa.py`
  is fitted on `train.parquet` participants with no check against
  `subject_split_across_traintest`. Treat subsample numbers as an upper bound
  until that is filtered.

**Known defect, open for decision — sparse-cadence rows fall back to a
constant.** 2,562 live rows get no window: all 281 `ShanghaiT1DM` rows (CGM every
15 min, so 33% of the 5-minute grid, below `min_valid_frac` 0.5) and 2,281 of
19,530 `BrisT1D` rows (15-minute stretches). CGM(t) is observed on every one of
them, yet they are predicted as the population median (131 mg/dL) at every
horizon: RMSE ~75 on those rows versus 32.9 for last-value persistence at 30 min.
Costs ~0.1 mg/dL of overall 30-min RMSE, ~0 at 120 min. The persistence
reference row uses the same fallback, so the JEPA-vs-persistence gap is
unaffected. `docs/2026-09-23/laptop-runner-and-lean-memory.md` §6.

**Known limitation of the JEPA baseline, by construction:** the published
CGM-JEPA checkpoints are an *encoder only*. The pretraining objective is an L1
loss between predicted and EMA-target patch embeddings in 96-d latent space; the
JEPA predictor was never released and there is no decoder from latent space back
to mg/dL. A forecast from these weights is therefore impossible without a fitted
readout. `predict_jepa.py` fits the smallest defensible one — a closed-form
ridge on frozen embeddings — and always logs a persistence row on the identical
rows, because the gap between those two rows is the only honest measure of what
the representation contributed.

## 5. Next steps

- **Fine-tune comparison done (200 k subsample, overall RMSE, one seed):**
  `pretrained` 27.38 (16.19 / 25.04 / 30.47 / 34.38), `frozen` 27.53, `scratch`
  27.35 — within 0.2 of each other, i.e. neither JEPA pretraining nor fine-tuning
  helps at this data size; val plateaus by epoch ~11 while train keeps falling.
  Next: submit one (`--checkpoint … --predict-full`), then participant-level
  context (test records span a median 196 days per participant; 33% of scored
  rows are participants also in train). `docs/2026-09-23/finetune-jepa.md`.
- **Covariate probe results (200 k subsample, pooled overall RMSE).** JEPA-only
  ridge 37.36 → + covariates, gbm 30.49 → `rich` gbm 29.3 (48 k fit windows) →
  `rich` gbm2000 28.08 / `richnojepa` gbm2000 **27.90** (506 k fit windows;
  16.59 / 25.82 / 31.30 / 34.52). Frozen JEPA features add nothing once
  covariates are present. Next: a full-template GBM driver for a real submission.
  `docs/2026-09-23/covariate-probe-rich.md`.
- **Fix the sparse-cadence fallback** (last-value persistence) under a new tag
  before submitting. `docs/2026-09-23/laptop-runner-and-lean-memory.md` §6.
- **Bring the cluster up before fanning out.** Install `requirements.txt` +
  `requirements-jepa.txt` into the `glucose-pred` env, then run ONE job and watch
  it — not four into an unverified partition.
  `docs/2026-09-20/cluster-scripts-and-streaming-eval.md`.
- **Exclude `subject_split_across_traintest` participants from the fit set**, or
  at minimum report the metric split by that flag. Same doc.
- **Recover `data/annual_competition/template.parquet`** and decide whether to
  target the annual competition; the deadline is 2026-10-05 23:59 AoE. Same doc.
- **Decide the fate of `model.py` / `train.py`.** They are the only path in this
  repo to a model that uses insulin and carbs, which is what the test set was
  filtered to reward, and they have no trained checkpoint.

Add an entry here when a doc leaves work open; delete it when the work lands.

### Orientation reading order

Newest first. A cold session should read in this order:

1. **This file, §4 and §5** — what is current, what is broken, what is untried.
2. **`EXPERIMENTS.md`** — the present spec: data, experiments, hyperparameters,
   outputs. Check its "Last synchronized with code" date before trusting it.
3. **`docs/2026-09-23/finetune-jepa.md`** — end-to-end fine-tuning design,
   controls, and what was verified (no real-data run yet).
4. **`docs/2026-09-23/covariate-probe-rich.md`** — round 2: 5-min dose slots,
   dose-action features, CGM summaries; `rich` / `richnojepa` variants.
5. **`docs/2026-09-23/covariate-probe.md`** — the insulin/carb probe, its
   feature bins, and why future inputs are an upper bound on AID data.
6. **`docs/2026-09-23/laptop-runner-and-lean-memory.md`** — the laptop driver,
   the memory cuts, and the bit-for-bit equivalence tests behind them.
7. **`docs/2026-09-20/cluster-scripts-and-streaming-eval.md`** — the cluster
   layout, the streaming eval path, and the allocation. Predates any
   completed Slurm run.
8. **`docs/2026-09-20/cgm-jepa-zero-shot-baseline.md`** — why the JEPA baseline
   is shaped the way it is, and the judgment calls behind it. Predates any
   `--scope full` run.
9. **`README.md`** — upstream's, and the one file written for an outside
   reader. Authoritative on the submission contract; silent on everything this
   fork added. It must be updated alongside `EXPERIMENTS.md` whenever a protocol
   changes: a reviewer reading a stale reproduction command is a worse failure
   than a stale internal note.
