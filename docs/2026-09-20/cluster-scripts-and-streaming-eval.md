# Cluster scripts, and a streaming eval path to make a full run possible

## 1. What changed

**New cluster scripts** (all additive, modelled on `CCQ-Dataset-Experiments`)

- `sbatch_overrides.sh` — sourced by the fan-out; turns
  `ACCOUNT`/`QOS`/`PARTITION`/`CPUS`/`MEM`/`WALLTIME` into sbatch CLI flags that
  beat the driver header. `mkdir -p logs/sbatch`. No `GPUS` knob.
- `run_jepa_cluster.sh` — the sbatch driver. Required, format-checked `--date`;
  fail-fast validation of `data/` and `weights/` before any work; per-config log
  redirection into `logs/<date>/jepa_zeroshot/`; phases `sub` then `full`, with
  a failing phase recorded rather than aborting (`STOP_ON_FAIL=1` to change).
- `submit_jepa_all.sh` — fan-out, one job per (encoder × fill). Three
  preflights: data present (with a size/mtime table), weights present, and the
  target tree holds no result CSVs (`ALLOW_EXISTING=1` to override).
  `SHARD_BY_SOURCE=1` splits the full phase one job per source dataset.
- `upload_to_cluster.sh` — allow-list code push.
- `upload_data_to_cluster.sh` — `data/` push, with a template/targets row-count guard.
- `download_from_cluster.sh` — one date's `results/` + `logs/` down, excluding
  `.*.lock` sidecars, then prints row counts and any `FAILED` rows.
- `assemble_submission.py` — merges per-shard prediction parquets onto the
  template and writes a generated `PROVENANCE.md`.

**Modified**

- `jepa_windows.py` — `build_eval_set` replaced by `iter_eval_chunks`, a
  bounded-memory generator. `_grid` renamed `grid_of` (now public, called once
  per participant); `extract` split into `extract_from_grid` plus a thin
  frame-level wrapper.
- `predict_jepa.py` — consumes the generator and predicts chunk by chunk; new
  `--only-source` (shard) and `--eval-chunk`; the shard is spliced into the
  method tag and the scope tag.

**Unchanged:** `run.py`, `metrics.py`, `model.py`, `train.py`,
`prepare_windows.py`, `jepa_model.py`, `requirements.txt`, `data/`. Nothing in
`CCQ-Dataset-Experiments` was touched — it was read only.

## 2. Why

Victor's ask: *"This folder includes a previous project i worked on which
involved cluster use. I want you to read through it (do not change anything) and
then add the scripts for uploading, running, and downloading from the cluster
for my full experiment run."*

Reading the CCQ scripts first surfaced a blocker in our own code. `--scope full`
could not have run on any allocation: `build_eval_set` materialized every
history and then every patch embedding before predicting anything. At 2.65 M
rows that is ~3 GB of histories and **~24 GB of embeddings**. The failure mode
would have been a cgroup SIGKILL, which — as CCQ's own `CLAUDE.md` records —
surfaces as *missing* rows rather than FAILED ones: a tree that looks clean and
is half empty. Fixing that had to come before any sbatch header.

## 3. Decisions made

- **Stream the eval path rather than shard around the memory problem.**
  Sharding by source would have capped a job's size, but `Loop` alone is 77% of
  the live template, so the biggest shard would still have been 2 M rows.
  Chunking fixes it at the root and makes peak memory independent of row count,
  which is why one modest allocation now covers the whole template. Sharding
  stayed as an *optional* wall-clock knob, off by default.
- **One code path for the subsample and the full run.** `iter_eval_chunks`
  replaced `build_eval_set` rather than sitting beside it, so a laptop
  subsample and a cluster full pass cannot diverge in windowing behavior. The
  refactor is result-neutral and was verified so: the 20 k subsample reproduces
  21.17 / 34.90 / 42.36 / 46.57 RMSE and 19,973 windowed rows exactly.
- **`grid_of` hoisted out of the chunk loop.** `extract` used to sort and
  de-duplicate the whole participant frame on every call; calling it per chunk
  would have made a chunked pass quadratic in participant length. Splitting out
  `extract_from_grid` keeps it once per participant.
- **Fan out by configuration, not by state-equivalent.** CCQ's unit of
  parallelism is the state because each state is a separate model fit. Here the
  four (encoder × fill) configurations are the independent units; sources are
  not, because the ridge head is fitted once across all of them. So the default
  is four jobs, and `SHARD_BY_SOURCE` exists for when wall clock matters.
- **The subsample phase runs in exactly one shard.** It is a whole-mix metric
  over the leaderboard's source proportions; a per-source slice of it would not
  be comparable to the full pass or to another shard. Under `SHARD_BY_SOURCE=1`
  the fan-out sets `RUN_SUB=0` on every shard but the first.
- **Sub before full.** The subsample costs well under a minute and is the only
  phase that yields a metric, so a misconfigured run announces itself before the
  long phase starts.
- **`weights/` ships with the code push** — the one directory that does. It is
  8 MB, and compute nodes frequently have no route to the Hugging Face hub, so
  fetching at job start is not dependable. CCQ's allow-list transfers no
  directory at all; this is a deliberate, documented departure.
- **Allow-list, not block-list, copied wholesale.** CCQ moved to one on
  2026-09-13 after a block-list quietly pushed 378 MB of re-identifiable data to
  shared storage, because rsync never reads `.gitignore`. The same hazard exists
  here: `data/` is 1.5 GB of clinical CGM under a `*.parquet` gitignore that the
  push would not have honoured.
- **Overlapping shards are an error, not a merge.** Two shards covering one row
  means the fan-out was misconfigured, and the quieter of two answers is not
  obviously the right one. Same reasoning for refusing to merge across
  configurations without `--tag`.
- **Merge by key, never by position.** A shard's rows are a subset in its own
  order; aligning positionally would silently transpose predictions between
  participants — a wrong answer with no error.
- **Allocation sized from measurement.** 8 CPUs / 16G / 4h. The encoder
  sustains ~2,100 windows/s/core (measured: 4,235 win/s on 2 cores), so the
  compute is minutes and the wall clock is dominated by reading `test.parquet`.
  Peak RSS is 1.3 GB at `--eval-chunk 50000`, so 16G is ~10× headroom. Walltime
  is left generous because CCQ's cluster has `PriorityWeightJobSize=0` — a
  longer walltime costs nothing in priority and a short one kills a real run.

**Verification performed.** Six shell scripts pass `bash -n`; three Python
modules compile. End to end on the synthetic fixture: two shards written, the
assembler merged them to 100% coverage, and the assembled file passed the
competition's own `run.py --competition live --horizon all`. Guards exercised
with exit codes checked — incomplete coverage exits 1 and names the gap by
source, `--check` on the same tree exits 0, overlapping shards exit 1, two
configurations without `--tag` exit 1. Fan-out dry-runs confirm four jobs by
default, 9 per config under `SHARD_BY_SOURCE=1` with `RUN_SUB=1` on the first
shard only, and a refusal on a tree that already holds results. Driver guards
fire for a missing `--date`, a malformed date, missing data, and missing weights.

## 4. What did NOT change / out of scope

- **`CCQ-Dataset-Experiments` was read only.** No file there was modified.
- **Nothing has been submitted to a cluster.** The scripts have never been run
  against a real Slurm scheduler. `GP_REMOTE_DIR` defaults to
  `/users/vmli3/glucose-pred`, which does not exist yet.
- **`--scope full` has still never completed on real data.** It was launched on
  the laptop but the device bridge dropped mid-run; the measurements above come
  from a synthetic-window benchmark of the dominant cost, not from a full pass.
- **No conda environment exists on the cluster.** `GP_ENV` defaults to
  `glucose-pred`; it has to be created from `requirements.txt` +
  `requirements-jepa.txt` before the first job.
- **The partition and account are inherited assumptions**, copied from CCQ's
  CPU-only SHAP driver (`oxf-c64-m512`, `--account=general`). Unverified for
  this project.
- **The annual competition is still untouched** — template missing, `-2`/`-1`
  sentinels unhandled.
- **`subject_split_across_traintest` is still unfiltered.**

## 5. Later the same day: GPU support and data on scratch

Victor: *"why is it cpu only? is gpu not needed? even if not needed, lets utilize
this node since the cpu nodes are all full: rp6b-1-gm96-c8-m64"* and *"also move
the data to be stored on scratch"*.

- **GPU is optional, not required, and the framing was wrong.** "CPU-only" was
  argued from the model being small, which is true and beside the point: what
  decides wall clock on a busy cluster is queue time, and an idle GPU node beats
  a full CPU partition however little of the card gets used. `--device auto` now
  picks up a card when one is allocated; the driver header still requests none,
  so a GPU stays an explicit choice via `GPUS=1 PARTITION=rp6b-1-gm96-c8-m64`.
- **TF32 is disabled explicitly.** On Ampere and later it turns float32 matmuls
  into 10-bit-mantissa operations, which would make the same window embed
  differently depending on which node the job landed on. The speed is irrelevant
  for a 522 k-parameter encoder; the reproducibility is not.
- **`check_device_parity` runs at startup on any accelerator.** It embeds a
  fixed batch on CPU and on the device and raises if they differ by more than
  2e-4. A silent numerical divergence does not fail a run, it just makes two
  results files incomparable — exactly the failure that is cheapest to catch at
  second zero and most expensive to catch in analysis.
- **Batch size defaults by device** (512 CPU / 4096 GPU) rather than being one
  number that is wrong on one of them.
- **Data moved to scratch**, `GP_REMOTE_DATA_DIR`, default
  `/scratch/<user>/glucose-pred/data`. Home directories are quota'd and backed
  up; 1.5 GB of parquet belongs on the large unbacked-up filesystem.
  `upload_data_to_cluster.sh` symlinks `<code tree>/data` at it, so `DATA_DIR`
  keeps its plain `data` default and no driver had to learn a second path.
  `ln -sfn`, not `ln -sf` — the latter nests the link inside its own target on
  the second run and leaves you with `data/data`. The script refuses to replace
  a real directory rather than deleting someone's files on its own judgment.
- **Scratch purges are now a named failure mode** in both drivers' comments and
  in `CLAUDE.md`, because a fail-fast guard tripping on a tree that ran last
  month looks like a code fault and is not.

Verified: `--device cpu` and `--device auto` produce identical metrics on the
fixture; `resolve_device("cuda")` raises cleanly with no card; the parity check
is a no-op on CPU; the GPU fan-out dry-run emits the right sbatch flags; a
symlinked `data` is followed and reported with its target, and a dangling one
fails the preflight rather than a job. The CUDA path itself is **unexercised** —
no GPU was available to test on.

## 6. Follow-ups

- **Create the cluster environment and run one job before the sweep.**
  `upload_to_cluster.sh`, `upload_data_to_cluster.sh`, then a single
  `ENCODERS=x_cgm_jepa FILLS=interp bash submit_jepa_all.sh --date <D>` and
  watch it, rather than fanning out four jobs into an unverified partition.
- **Confirm the partition/account pair.** `sacctmgr -nP show assoc
  user=$USER format=account,partition` — a mismatch is rejected at submit time.
- **Complete a `--scope full` pass and validate it with `run.py`.** Still the
  gate before any submission; the Oct 5 annual deadline is the clock.
- **Check `sacct MaxRSS` on the first full job.** The 1.3 GB figure is a
  synthetic-window benchmark; a real participant frame from `test.parquet` is
  larger than anything that benchmark held.
- **Consider a `--only-source` smoke cell first** — `ONLY_SOURCE=ShanghaiT1DM`
  is 281 rows and exercises the whole path in seconds.
- **Watch the first GPU job's parity line.** `check_device_parity` is unexercised
  on real hardware; if it raises, the run is correct to stop and `--device cpu`
  is the fallback.
- **Confirm the scratch path and its purge policy** before relying on it —
  `/scratch/vmli3/` is inferred from the CCQ repo's Qwen checkpoint path, not
  verified for this account.
