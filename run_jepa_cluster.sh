#!/bin/bash
#SBATCH --job-name=jepa_zeroshot
#SBATCH --account=general
#SBATCH --nodes=1
#SBATCH --partition=oxf-c64-m512
#SBATCH --output=logs/sbatch/%x_%j.out
#SBATCH --error=logs/sbatch/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# =============================================================================
# jepa_zeroshot -- frozen CGM-JEPA encoder + closed-form ridge, on the cluster
#
# =============================================================================
# Runs ONE configuration (encoder x fill) in two phases:
#
#   sub   a stratified subsample scored against the live targets, with a
#         persistence row on the identical rows. This is the diagnostic: it is
#         what tells you whether the configuration is worth the full pass.
#   full  every template row, written as a submission parquet.
#
# SUB RUNS FIRST on purpose. It costs well under a minute and it is the only
# phase that produces a metric, so a misconfigured run announces itself before
# the long phase starts. A failing phase does not abort the job (set
# STOP_ON_FAIL=1 to change that); the job exits non-zero and the summary names
# which phase failed.
#
# RUNS ON EITHER. The encoder is 522 k parameters, so nothing here NEEDS a GPU
# and the default header requests none. But a GPU is frequently the faster way
# to finish anyway, because queue time dominates the run: an idle GPU node beats
# a full CPU partition however little of the card the job uses. Ask for one with
#   GPUS=1 PARTITION=rp6b-1-gm96-c8-m64 MEM=48G
# and DEVICE=auto picks it up. Nothing else changes -- the ridge solve is float64
# numpy on the host either way, and only the encoder forward pass moves.
#
# The script reads SLURM_CPUS_PER_TASK for the torch thread count rather than
# grabbing every visible core on a shared node. That still matters on a GPU
# node: windowing, gap-filling and the ridge are all host-side, and on the full
# template they are the larger half of the wall clock.
#
# ALLOCATION, from measurement rather than guesswork. The encoder sustains
# ~2,100 windows/s/core, so the full 2.65 M-row live template is a few minutes
# of compute on 8 cores and the wall clock is dominated by reading test.parquet,
# not by the model. Peak RSS on the full live template is ~1.5 GB at the default
# --eval-chunk of 50,000, measured on real data, because the eval path streams:
# per-chunk arrays are the histories (58 MB) and the features (77 MB), patch
# embeddings are reduced to features one forward batch at a time, and the only
# arrays that scale with the row count are the predictions, anchors, origins and
# the categorical template keys (~150 MB at full size). 16G is therefore about
# ten times headroom, and EVAL_CHUNK is the knob if a node is tighter than that;
# it changes no result.
#
# FAN-OUT: every output, results CSV and log is per-configuration, so
# configurations run as SEPARATE CONCURRENT JOBS. submit_jepa_all.sh is the
# intended entry point. SHARD_BY_SOURCE splits the full phase further, one job
# per source dataset -- only worth it if wall clock matters, since a whole-set
# pass is bounded by chunked memory rather than by row count.
#
# Knobs (env, all optional):
#   ENCODER        x_cgm_jepa (default) | cgm_jepa
#   FILL           interp (default) | sentinel
#   COMPETITION    live (default) | annual
#   RUN_SUB        1 = run the scored subsample phase (default 1)
#   RUN_FULL       1 = run the submission phase (default 1)
#   EVAL_ROWS      subsample size for the sub phase (default 200000)
#   ONLY_SOURCE    restrict the FULL phase to these source datasets (a shard)
#   EVAL_CHUNK     windows embedded per batch; sets peak memory (default 50000)
#   DEVICE         auto (default) | cpu | cuda. auto uses a GPU when one is visible
#   BATCH_SIZE     windows per forward pass (default 512 on CPU, 4096 on GPU)
#   FIT_ROW_GROUPS train.parquet row groups to read (default 24)
#   FIT_PER_GROUP  max fit windows per participant (default 150)
#   DATA_DIR       default `data`
#   GP_ENV         conda env (default glucose-pred)
#   DRY_RUN        1 = print the commands, run nothing
#
# Usage:
#   sbatch run_jepa_cluster.sh --date YYYY-MM-DD
#   ENCODER=cgm_jepa FILL=sentinel sbatch run_jepa_cluster.sh --date YYYY-MM-DD
#
# AFTER a sharded full run, reassemble before validating:
#   python assemble_submission.py --date <date>
#   python run.py results/<date>/jepa_zeroshot/preds/<...>_assembled.parquet \
#       --competition live --horizon all
# =============================================================================

conda init bash > /dev/null 2>&1
source ~/.bashrc

DATA_DIR="${DATA_DIR:-data}"
GP_ENV="${GP_ENV:-glucose-pred}"
PY=(conda run -n "$GP_ENV" python -u)

# Keep torch and BLAS from oversubscribing a shared CPU allocation. Left alone,
# torch sizes its thread pool from the machine's core count, not the cgroup's,
# which on a 64-core node means 64 threads inside an 8-core allocation.
NTHREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NTHREADS}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TORCH_NUM_THREADS="$OMP_NUM_THREADS"

RESULTS_DATE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --date)   RESULTS_DATE="$2"; shift 2 ;;
        --date=*) RESULTS_DATE="${1#*=}"; shift ;;
        *) echo "Unknown argument: $1 (only --date YYYY-MM-DD is accepted)" >&2; exit 1 ;;
    esac
done
if [ -z "$RESULTS_DATE" ]; then
    echo "ERROR: --date YYYY-MM-DD is required, e.g. sbatch $0 --date $(date +%F)" >&2
    exit 1
fi
if ! [[ "$RESULTS_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "ERROR: --date must be YYYY-MM-DD, got '$RESULTS_DATE'." >&2
    exit 1
fi

ENCODER="${ENCODER:-x_cgm_jepa}"
FILL="${FILL:-interp}"
COMPETITION="${COMPETITION:-live}"
RUN_SUB="${RUN_SUB:-1}"
RUN_FULL="${RUN_FULL:-1}"
EVAL_ROWS="${EVAL_ROWS:-200000}"
EVAL_CHUNK="${EVAL_CHUNK:-50000}"
FIT_ROW_GROUPS="${FIT_ROW_GROUPS:-24}"
FIT_PER_GROUP="${FIT_PER_GROUP:-150}"
WEIGHTS="${WEIGHTS:-weights/cgm_jepa_hf}"
DEVICE="${DEVICE:-auto}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Fail fast on inputs BEFORE doing any work. data/ is gitignored and is NOT part
# of the code push, so a cluster copy can silently be absent or stale; a sweep
# that dies an hour in on a missing parquet costs the whole allocation.
#
# On the cluster DATA_DIR is a SYMLINK into scratch (upload_data_to_cluster.sh
# creates it). Scratch is typically purged on a schedule, so a guard that trips
# on a tree that worked last month usually means the sweep took the data with
# it -- re-run the upload rather than hunting for a code fault. The checks below
# follow the link, so a dangling one reports as missing files.
COMP_DIR="${DATA_DIR}/live_leaderboard"
[ "$COMPETITION" = "annual" ] && COMP_DIR="${DATA_DIR}/annual_competition"

# Say WHAT is wrong with DATA_DIR, not just that it is empty. On the cluster
# data/ is a symlink into scratch, so the three realistic causes -- never
# pushed, scratch purged, link missing -- look identical from a file-existence
# check and need different fixes.
data_dir_diagnosis() {
    local d="$1" scratch="${GP_REMOTE_DATA_DIR:-/scratch/${USER:-\$USER}/glucose-pred/data}"
    if [ -L "$d" ]; then
        local tgt; tgt="$(readlink "$d")"
        if [ -d "$tgt" ]; then
            echo "       '$d' is a symlink -> $tgt, which exists but does not hold those files." >&2
        else
            echo "       '$d' is a DANGLING symlink -> $tgt." >&2
            echo "       Scratch is purged on a schedule; that is the usual cause." >&2
        fi
    elif [ -d "$d" ]; then
        echo "       '$d' is a real directory, not a symlink into scratch, and it is empty." >&2
    else
        echo "       '$d' does not exist." >&2
    fi
    echo "       Fixes, likeliest first:" >&2
    echo "         1. push the data (run on your laptop, not here):" >&2
    echo "              ./upload_data_to_cluster.sh" >&2
    echo "         2. if scratch already holds a copy, just relink:" >&2
    echo "              ln -sfn $scratch $d" >&2
    echo "         3. skip the link entirely for one run:" >&2
    echo "              DATA_DIR=$scratch <your command>" >&2
}

missing=()
for f in "${DATA_DIR}/train.parquet" "${DATA_DIR}/test.parquet" "${COMP_DIR}/template.parquet"; do
    [ -f "$f" ] || missing+=("$f")
done
if [ "$RUN_SUB" = "1" ] && [ "$COMPETITION" = "live" ]; then
    [ -f "${COMP_DIR}/targets.parquet" ] || missing+=("${COMP_DIR}/targets.parquet")
fi
if [ ${#missing[@]} -gt 0 ]; then
    echo "ERROR: data not found under DATA_DIR='${DATA_DIR}' (cwd: $(pwd)). Missing:" >&2
    printf '         %s\n' "${missing[@]}" >&2
    data_dir_diagnosis "$DATA_DIR"
    exit 1
fi
if [ ! -d "${WEIGHTS}/${ENCODER}" ]; then
    echo "ERROR: encoder weights not found at ${WEIGHTS}/${ENCODER} (cwd: $(pwd))." >&2
    echo "       They ride along with ./upload_to_cluster.sh; if the compute nodes" >&2
    echo "       can reach the hub, fetch them instead with:" >&2
    echo "         huggingface-cli download CRUISEResearchGroup/CGM-JEPA \\" >&2
    echo "           --local-dir ${WEIGHTS} --include 'cgm_jepa/*' 'x_cgm_jepa/*'" >&2
    exit 1
fi

LOG_DIR="logs/${RESULTS_DATE}/jepa_zeroshot"
mkdir -p "$LOG_DIR" "results/${RESULTS_DATE}/jepa_zeroshot"

# ONLY_SOURCE may name several datasets ("Loop ReplaceBG"), which would put a
# SPACE in the log filename.
SHARD_TAG=$(printf '%s' "${ONLY_SOURCE:-all}" | tr -s '[:space:],' '-' | tr -cd '[:alnum:]-')
CFG_TAG="${ENCODER}_${FILL}_${SHARD_TAG}"
exec >>"${LOG_DIR}/${CFG_TAG}_${SLURM_JOB_ID:-manual}.out" \
    2>>"${LOG_DIR}/${CFG_TAG}_${SLURM_JOB_ID:-manual}.err"

run() { if [ "$DRY_RUN" = "1" ]; then echo "[dry-run] $*"; else "$@"; fi; }

echo "[jepa] date=$RESULTS_DATE encoder=$ENCODER fill=$FILL competition=$COMPETITION"
echo "[jepa] threads=$NTHREADS  device=$DEVICE  eval_chunk=$EVAL_CHUNK"
echo "[jepa] shard=${ONLY_SOURCE:-<whole template>}  gpus_allocated=${SLURM_GPUS:-${SLURM_JOB_GPUS:-none}}"
echo "[jepa] phases: sub=$RUN_SUB full=$RUN_FULL"
echo "[jepa] started $(date '+%F %T')"

SHARD_ARG=()
[ -n "${ONLY_SOURCE:-}" ] && SHARD_ARG=(--only-source "$ONLY_SOURCE")

FAILED=()

phase() {   # $1 = phase name, rest = extra args to predict_jepa.py
    local name="$1"; shift
    local t0 rc
    echo; echo "======== phase: ${name} ========"
    t0=$(date +%s)
    run "${PY[@]}" predict_jepa.py \
        --date "$RESULTS_DATE" \
        --competition "$COMPETITION" \
        --encoder "$ENCODER" \
        --fill "$FILL" \
        --weights "$WEIGHTS" \
        --data-dir "$DATA_DIR" \
        --eval-chunk "$EVAL_CHUNK" \
        --fit-row-groups "$FIT_ROW_GROUPS" \
        --fit-per-group "$FIT_PER_GROUP" \
        --device "$DEVICE" \
        ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} \
        "$@"
    rc=$?
    echo "-------- ${name}: rc=${rc} in $(( $(date +%s) - t0 ))s --------"
    if [ "$rc" -ne 0 ]; then
        FAILED+=("$name")
        [ "$STOP_ON_FAIL" = "1" ] && { echo "STOP_ON_FAIL=1 -- aborting."; exit "$rc"; }
    fi
    return 0
}

# The subsample never shards: its whole point is a metric over the leaderboard's
# source mix, and a per-source slice of it would not be comparable to the full
# pass or to another shard.
if [ "$RUN_SUB" = "1" ]; then
    phase "sub" --scope sub --eval-rows "$EVAL_ROWS" --sample-mode proportional --quiet
fi

if [ "$RUN_FULL" = "1" ]; then
    phase "full" --scope full "${SHARD_ARG[@]}" --quiet
fi

echo
echo "[jepa] finished $(date '+%F %T')"
echo "[jepa] results -> results/${RESULTS_DATE}/jepa_zeroshot/"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[jepa] FAILED phases: ${FAILED[*]}"
    exit 1
fi
echo "[jepa] all phases ok"
