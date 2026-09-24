#!/bin/bash
# =============================================================================
# run_jepa_local.sh -- the jepa_zeroshot sweep on a laptop, no scheduler
#
# Runs every (encoder x fill) configuration ONE AFTER ANOTHER in this shell:
# sub (scored subsample), then full (template-complete parquet), then run.py
# validation of the full parquet. Sequential on purpose: each configuration
# already uses every performance core for the encoder, so running two at once
# on one machine only makes them contend.
#
# MEMORY. Peak RSS is ~1 GB for either phase, and it does not grow with the
# number of rows scored: the eval path streams in chunks of EVAL_CHUNK windows,
# the template is held as categorical keys, and patch embeddings are reduced to
# features batch by batch. EVAL_CHUNK defaults lower here than on the cluster;
# it trades a little speed for a smaller working set and changes no result.
# Each run's high-water mark is printed and logged as peak_rss_gb in the run
# config JSON, so the figure is measured on this machine, not assumed.
#
# SLEEP. A full pass takes minutes per configuration. On macOS the script
# re-launches itself under `caffeinate -i` so idle sleep cannot suspend it. That
# does not stop the lid from sleeping the machine on battery: keep it open, or on
# power.
#
# Knobs (env, all optional):
#   PYTHON         interpreter with requirements.txt + requirements-jepa.txt
#                  installed (default: python3 on PATH -- activate the env first)
#   ENCODERS       space-separated (default "x_cgm_jepa cgm_jepa")
#   FILLS          space-separated (default "interp sentinel")
#   COMPETITION    live (default) | annual
#   RUN_SUB        1 = run the scored subsample phase (default 1)
#   RUN_FULL       1 = run the submission phase (default 1)
#   VALIDATE       1 = run.py on each full parquet (default 1; live only scores)
#   EVAL_ROWS      subsample size for the sub phase (default 200000)
#   ONLY_SOURCE    restrict the FULL phase to these source datasets (a shard;
#                  e.g. ShanghaiT1DM is 281 rows -- a seconds-long smoke test)
#   EVAL_CHUNK     windows per streamed chunk (default 10000)
#   BATCH_SIZE     windows per forward pass (default 512)
#   THREADS        torch/BLAS threads (default: performance-core count)
#   FIT_ROW_GROUPS train.parquet row groups to read (default 24)
#   FIT_PER_GROUP  max fit windows per participant (default 150)
#   DATA_DIR       default `data`
#   ALLOW_EXISTING 1 = re-run phases whose prediction parquet already exists
#                  (default 0: such phases are skipped, so a sweep started one
#                  configuration at a time finishes by re-running the same command)
#   STOP_ON_FAIL   1 = abort at the first failing phase (default 0)
#   DRY_RUN        1 = print the commands, run nothing
#
# Usage:
#   bash run_jepa_local.sh --date YYYY-MM-DD
#   ENCODERS=x_cgm_jepa FILLS=interp bash run_jepa_local.sh --date YYYY-MM-DD
#   RUN_SUB=0 VALIDATE=0 ONLY_SOURCE=ShanghaiT1DM bash run_jepa_local.sh --date YYYY-MM-DD
#
# Written for the bash 3.2 that ships with macOS: no associative arrays, no
# mapfile, no ${var,,}.
# =============================================================================

cd "$(dirname "$0")" || exit 1

if [ -z "${GP_CAFFEINATED:-}" ] && command -v caffeinate > /dev/null 2>&1 && [ "${DRY_RUN:-0}" != "1" ]; then
    exec caffeinate -i env GP_CAFFEINATED=1 bash "$0" "$@"
fi

RESULTS_DATE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --date)   RESULTS_DATE="$2"; shift 2 ;;
        --date=*) RESULTS_DATE="${1#*=}"; shift ;;
        *) echo "Unknown argument: $1 (only --date YYYY-MM-DD is accepted)" >&2; exit 1 ;;
    esac
done
if [ -z "$RESULTS_DATE" ]; then
    echo "ERROR: --date YYYY-MM-DD is required, e.g. bash $0 --date $(date +%F)" >&2
    exit 1
fi
if ! [[ "$RESULTS_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "ERROR: --date must be YYYY-MM-DD, got '$RESULTS_DATE'." >&2
    exit 1
fi

PYTHON="${PYTHON:-python3}"
ENCODERS="${ENCODERS:-x_cgm_jepa cgm_jepa}"
FILLS="${FILLS:-interp sentinel}"
COMPETITION="${COMPETITION:-live}"
RUN_SUB="${RUN_SUB:-1}"
RUN_FULL="${RUN_FULL:-1}"
VALIDATE="${VALIDATE:-1}"
EVAL_ROWS="${EVAL_ROWS:-200000}"
EVAL_CHUNK="${EVAL_CHUNK:-10000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
FIT_ROW_GROUPS="${FIT_ROW_GROUPS:-24}"
FIT_PER_GROUP="${FIT_PER_GROUP:-150}"
WEIGHTS="${WEIGHTS:-weights/cgm_jepa_hf}"
DATA_DIR="${DATA_DIR:-data}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Performance cores only. Apple silicon pairs P-cores with slower E-cores, and a
# thread pool sized to all of them waits on the E-cores at every sync point.
if [ -z "${THREADS:-}" ]; then
    THREADS=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null \
              || sysctl -n hw.physicalcpu 2>/dev/null \
              || nproc 2>/dev/null || echo 4)
fi
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
export TORCH_NUM_THREADS="$THREADS"

# --- preflight: environment, data, weights, results tree -------------------
# The interpreter check comes first: a missing package otherwise fails every
# phase within seconds and leaves no results row to say why.
MISSING_PKGS=$("$PYTHON" -c "
import importlib.util as u
print(' '.join(m for m in ('numpy', 'pandas', 'pyarrow', 'torch', 'safetensors') if u.find_spec(m) is None))
" 2>/dev/null) || MISSING_PKGS="(could not run '$PYTHON')"
if [ -n "$MISSING_PKGS" ]; then
    echo "ERROR: '$PYTHON' ($(command -v "$PYTHON")) is missing: $MISSING_PKGS" >&2
    echo "       Install into THAT interpreter (python -m pip, not a bare pip):" >&2
    echo "         $PYTHON -m pip install -r requirements.txt -r requirements-jepa.txt" >&2
    echo "       or point PYTHON at an env that has them:" >&2
    echo "         PYTHON=/path/to/env/bin/python bash $0 --date $RESULTS_DATE" >&2
    exit 1
fi

COMP_DIR="${DATA_DIR}/live_leaderboard"
[ "$COMPETITION" = "annual" ] && COMP_DIR="${DATA_DIR}/annual_competition"
missing=()
for f in "${DATA_DIR}/train.parquet" "${DATA_DIR}/test.parquet" "${COMP_DIR}/template.parquet"; do
    [ -f "$f" ] || missing+=("$f")
done
if [ "$RUN_SUB" = "1" ] && [ "$COMPETITION" = "live" ]; then
    [ -f "${COMP_DIR}/targets.parquet" ] || missing+=("${COMP_DIR}/targets.parquet")
fi
if [ ${#missing[@]} -gt 0 ]; then
    echo "ERROR: data not found under DATA_DIR='${DATA_DIR}'. Missing:" >&2
    printf '         %s\n' "${missing[@]}" >&2
    exit 1
fi
for enc in $ENCODERS; do
    if [ ! -d "${WEIGHTS}/${enc}" ]; then
        echo "ERROR: encoder weights not found at ${WEIGHTS}/${enc}. Fetch with:" >&2
        echo "         huggingface-cli download CRUISEResearchGroup/CGM-JEPA \\" >&2
        echo "           --local-dir ${WEIGHTS} --include 'cgm_jepa/*' 'x_cgm_jepa/*'" >&2
        exit 1
    fi
done

RES_DIR="results/${RESULTS_DATE}/jepa_zeroshot"
LOG_DIR="logs/${RESULTS_DATE}/jepa_zeroshot"
[ "$DRY_RUN" = "1" ] || mkdir -p "$RES_DIR" "$LOG_DIR"

SHARD_ARG=()
SHARD_SUFFIX=""
if [ -n "${ONLY_SOURCE:-}" ]; then
    SHARD_ARG=(--only-source "$ONLY_SOURCE")
    # Must match the suffix predict_jepa.py puts in the scope tag.
    SHARD_SUFFIX="_$(printf '%s' "$ONLY_SOURCE" | tr -cs 'A-Za-z0-9' '-' | sed 's/^-//; s/-$//')"
fi
SHARD_TAG=$(printf '%s' "${ONLY_SOURCE:-all}" | tr -s '[:space:],' '-' | tr -cd '[:alnum:]-')
# Rows a full phase writes -- the whole template, or the shard's share of it --
# which is the <n> in its prediction filename.
N_FULL=$(ONLY_SOURCE="${ONLY_SOURCE:-}" "$PYTHON" -c "
import os, re, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
path = '${COMP_DIR}/template.parquet'
keep = [s for s in re.split(r'[,\s]+', os.environ['ONLY_SOURCE']) if s]
if not keep:
    print(pq.ParquetFile(path).metadata.num_rows)
else:
    sf = pq.read_table(path, columns=['source_file'])['source_file']
    print(pc.sum(pc.is_in(sf, value_set=pa.array(keep))).as_py() or 0)
")

# Result rows are append-only, so re-running a finished phase writes a second set
# of rows beside the first. A phase whose prediction parquet already exists is
# therefore skipped unless ALLOW_EXISTING=1.
already_done() {   # $1 = label, rest = candidate parquet paths (a glob may not match)
    local label="$1" f; shift
    [ "$ALLOW_EXISTING" = "1" ] && return 1
    for f in "$@"; do
        if [ -f "$f" ]; then
            echo; echo "======== ${label}: SKIPPED, ${f} exists (ALLOW_EXISTING=1 re-runs it) ========"
            return 0
        fi
    done
    return 1
}
STAMP=$(date +%H%M%S)

echo "[jepa-local] date=$RESULTS_DATE competition=$COMPETITION python=$(command -v "$PYTHON")"
echo "[jepa-local] encoders=[$ENCODERS] fills=[$FILLS] shard=$SHARD_TAG"
echo "[jepa-local] threads=$THREADS eval_chunk=$EVAL_CHUNK batch_size=$BATCH_SIZE"
echo "[jepa-local] phases: sub=$RUN_SUB full=$RUN_FULL validate=$VALIDATE"
echo "[jepa-local] started $(date '+%F %T')"

set -o pipefail
FAILED=()

run() {   # $1 = log file, rest = command
    local log="$1"; shift
    if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY_RUN: $*"
        return 0
    fi
    "$@" 2>&1 | tee -a "$log"
}

phase() {   # $1 = label, $2 = log file, rest = command
    local label="$1" log="$2"; shift 2
    local t0 rc
    echo; echo "======== ${label} ========" | tee -a "$log"
    t0=$(date +%s)
    run "$log" "$@"
    rc=$?
    echo "-------- ${label}: rc=${rc} in $(( $(date +%s) - t0 ))s --------" | tee -a "$log"
    if [ "$rc" -ne 0 ]; then
        FAILED+=("$label")
        if [ "$STOP_ON_FAIL" = "1" ]; then
            echo "STOP_ON_FAIL=1 -- aborting."
            exit "$rc"
        fi
    fi
    return "$rc"
}

for enc in $ENCODERS; do
    for fill in $FILLS; do
        cfg="${enc}_${fill}"
        log="${LOG_DIR}/${cfg}_${SHARD_TAG}_local-${STAMP}.out"
        [ "$DRY_RUN" = "1" ] && log=/dev/null
        common=("$PYTHON" -u predict_jepa.py
                --date "$RESULTS_DATE" --competition "$COMPETITION"
                --encoder "$enc" --fill "$fill" --weights "$WEIGHTS" --data-dir "$DATA_DIR"
                --eval-chunk "$EVAL_CHUNK" --batch-size "$BATCH_SIZE"
                --fit-row-groups "$FIT_ROW_GROUPS" --fit-per-group "$FIT_PER_GROUP"
                --device cpu --quiet)

        # The subsample never shards: it is a metric over the leaderboard's whole
        # source mix, and a per-source slice of it would not be comparable.
        stem="${RES_DIR}/preds/jepa_${enc}_ridge_${fill}_${COMPETITION}"
        if [ "$RUN_SUB" = "1" ] && ! already_done "${cfg} sub" "${stem}sub"[0-9]*.parquet; then
            phase "${cfg} sub" "$log" "${common[@]}" \
                --scope sub --eval-rows "$EVAL_ROWS" --sample-mode proportional
        fi

        if [ "$RUN_FULL" = "1" ]; then
            pred="${stem}full${N_FULL}${SHARD_SUFFIX}.parquet"
            full_ok=1
            if ! already_done "${cfg} full" "$pred"; then
                phase "${cfg} full" "$log" "${common[@]}" --scope full "${SHARD_ARG[@]}" || full_ok=0
            fi
            if [ "$full_ok" = "1" ] && [ "$VALIDATE" = "1" ] && [ -z "${ONLY_SOURCE:-}" ]; then
                phase "${cfg} validate" "$log" "$PYTHON" run.py "$pred" \
                    --competition "$COMPETITION" --horizon all
            fi
        fi
    done
done

echo
echo "[jepa-local] finished $(date '+%F %T')"
echo "[jepa-local] results -> ${RES_DIR}/"
echo "[jepa-local] logs    -> ${LOG_DIR}/"
[ -n "${ONLY_SOURCE:-}" ] && [ "$RUN_FULL" = "1" ] && \
    echo "[jepa-local] shard run: assemble before validating -- python assemble_submission.py --date $RESULTS_DATE"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[jepa-local] FAILED phases:"
    printf '    %s\n' "${FAILED[@]}"
    exit 1
fi
echo "[jepa-local] all phases ok"
