#!/bin/bash
# =============================================================================
# submit_jepa_all.sh -- fan out the full run, one job per configuration
# =============================================================================
# Submits one sbatch job per (encoder x fill) configuration. Each job runs that
# configuration's scored subsample and then its full submission pass. The work
# inside a job is sequential; the configurations are the parallelism.
#
#   bash submit_jepa_all.sh --date YYYY-MM-DD
#
# Use a FRESH --date for a full rerun. Result CSVs APPEND and never dedupe, so
# submitting twice into one tree gives you two rows per cell. The preflight below
# refuses a date whose tree already holds result CSVs unless ALLOW_EXISTING=1.
#
# Knobs (env, all optional):
#   ENCODERS         default "x_cgm_jepa cgm_jepa"
#   FILLS            default "interp sentinel"
#   COMPETITION      live (default) | annual
#   SHARD_BY_SOURCE  1 = split each configuration's FULL phase into one job per
#                    source dataset. Off by default: memory is bounded by
#                    --eval-chunk rather than by row count, so a whole-template
#                    pass fits in one modest allocation and sharding only buys
#                    wall clock. When on, the subsample phase runs in the FIRST
#                    shard only -- it is a whole-mix metric and a per-source
#                    slice of it is not comparable to anything.
#   SOURCES          which datasets to shard over (default: the nine in the live
#                    test set, largest first so the long pole starts first)
#   RUN_SUB / RUN_FULL     skip a phase in every submitted job
#   EVAL_ROWS, EVAL_CHUNK, FIT_ROW_GROUPS, FIT_PER_GROUP, DATA_DIR, GP_ENV
#   SKIP_ENV_CHECK   1 = do not probe the conda env before submitting
#   DRY_RUN          1 = print the sbatch lines, submit nothing
#   ALLOW_EXISTING   1 = submit into a tree that already has results
#   DEVICE           auto (default) | cpu | cuda, forwarded to the driver
#   ACCOUNT / QOS / PARTITION / CPUS / MEM / GPUS / WALLTIME
#                    from sbatch_overrides.sh; these BEAT the driver's #SBATCH
#                    header. ACCOUNT must match the partition -- check with
#                      sacctmgr -nP show assoc user=$USER format=account,partition
#
# Everything the driver reads is forwarded to every job via --export=ALL.
#
# ON A GPU NODE. Nothing here needs a GPU, but when the CPU partitions are full
# an idle GPU node is simply the faster queue:
#   GPUS=1 PARTITION=rp6b-1-gm96-c8-m64 MEM=48G CPUS=8 \
#     bash submit_jepa_all.sh --date YYYY-MM-DD
# MEM stays under the node's 64G. The driver's DEVICE=auto finds the card, and
# predict_jepa.py checks that it reproduces the CPU forward pass before using it.
#
# AFTER the jobs finish:
#   python assemble_submission.py --date <date> --tag <config>   # sharded runs only
#   python run.py results/<date>/jepa_zeroshot/preds/<file>.parquet \
#       --competition live --horizon all
# =============================================================================
set -euo pipefail

if [ "${1:-}" != "--date" ] || [ -z "${2:-}" ]; then
    echo "usage: bash $0 --date YYYY-MM-DD   (the results tree to write into)" >&2
    echo "       e.g. bash $0 --date $(date +%F)" >&2
    exit 1
fi
RESULTS_DATE="$2"
if ! [[ "$RESULTS_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "ERROR: --date must be YYYY-MM-DD, got '$RESULTS_DATE'." >&2
    exit 1
fi

# ---- must be the repo root -------------------------------------------------
for f in sbatch_overrides.sh run_jepa_cluster.sh predict_jepa.py jepa_model.py \
         jepa_windows.py metrics.py run.py; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found -- run this from the repo root (cwd='$PWD')." >&2
        exit 1
    fi
done

# ---- preflight 0: can the job's python import what it needs? ---------------
# Four jobs once burned an allocation apiece on a missing numpy, so this probes
# the environment before submitting. It resolves the env's interpreter from
# `conda env list` and runs it directly rather than going through `conda run`:
# conda run wraps the child, and its exit code and output reflect the wrapper as
# much as the command, which is not something a preflight can reason about.
#
# SEVERITY IS DELIBERATE. Only one outcome blocks: the interpreter ran and said a
# module is absent. Anything else -- env not listed, conda absent, probe failed
# -- WARNS and continues, printing whatever the probe actually said. A preflight
# that cannot tell "your environment is broken" from "my probe does not work
# here" must not be the thing that stops a submission; a false negative that
# blocks a working setup is worse than no check at all.
#
# SKIP_ENV_CHECK=1 skips it entirely.
GP_ENV="${GP_ENV:-glucose-pred}"
REQUIRED_MODULES="numpy pandas pyarrow torch safetensors"
if [ "${SKIP_ENV_CHECK:-0}" != "1" ]; then
    env_python=""
    if command -v conda >/dev/null 2>&1; then
        env_python=$(conda env list 2>/dev/null \
            | awk -v e="$GP_ENV" '$1==e {print $NF"/bin/python"}' | head -1)
    fi

    if [ -z "$env_python" ] || [ ! -x "$env_python" ]; then
        echo "NOTE: could not locate a python for conda env '$GP_ENV' from the login" >&2
        echo "      node, so its packages were not checked. Submitting anyway -- the" >&2
        echo "      jobs source ~/.bashrc and resolve the env themselves." >&2
    else
        # Assigned inside `if` so a failing probe cannot trip `set -e`: this is a
        # warning path, and a preflight must never be what kills the submission.
        if probe_out=$("$env_python" -c "
import importlib.util, sys
missing = [m for m in '${REQUIRED_MODULES}'.split() if importlib.util.find_spec(m) is None]
print('MISSING:' + ','.join(missing))
print('PY:' + sys.version.split()[0])" 2>&1); then probe_rc=0; else probe_rc=$?; fi
        env_missing=$(printf '%s\n' "$probe_out" | sed -n 's/^MISSING://p' | head -1)

        if [ "$probe_rc" -ne 0 ] || ! printf '%s\n' "$probe_out" | grep -q '^PY:'; then
            echo "NOTE: the environment probe did not run cleanly, so packages were not" >&2
            echo "      verified. Submitting anyway. It said:" >&2
            printf '        %s\n' "$probe_out" >&2
        elif [ -n "$env_missing" ]; then
            # The one unambiguous case: its own interpreter cannot find them.
            echo "ERROR: $env_python cannot import: ${env_missing//,/ }" >&2
            echo "       The env exists but those packages are not in it. The usual cause" >&2
            echo "       is 'conda activate' being a no-op in a non-interactive shell, so" >&2
            echo "       a following pip installed into a different python. Install BY" >&2
            echo "       NAME so it cannot go astray:" >&2
            echo "         conda run -n $GP_ENV python -m pip install \\" >&2
            echo "           'pandas<3.0.0' 'numpy>=1.24.0' 'pyarrow>=14.0.0' torch safetensors" >&2
            echo "       Override with SKIP_ENV_CHECK=1 if you believe this is wrong." >&2
            exit 1
        else
            echo "Environment '$GP_ENV': python $(printf '%s\n' "$probe_out" | sed -n 's/^PY://p'), all imports present."
        fi
    fi
fi

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

COMPETITION="${COMPETITION:-live}"
DATA_DIR="${DATA_DIR:-data}"
COMP_DIR="${DATA_DIR}/live_leaderboard"
[ "$COMPETITION" = "annual" ] && COMP_DIR="${DATA_DIR}/annual_competition"

# ---- preflight 1: the data is here, and it is the data you think it is -----
# data/ is gitignored and is NOT part of the code push, so a cluster copy can
# silently be absent or a previous export. The size/mtime table below is the
# cheapest way to confirm the right one landed: compare it against the same
# table locally. On the cluster data/ is a symlink into scratch; if scratch has
# been purged since the last run this is where you find out, before four jobs
# queue and then die on the same guard.
missing=()
for f in "${DATA_DIR}/train.parquet" "${DATA_DIR}/test.parquet" "${COMP_DIR}/template.parquet"; do
    [ -f "$f" ] || missing+=("$f")
done
if [ "$COMPETITION" = "live" ]; then
    [ -f "${COMP_DIR}/targets.parquet" ] || missing+=("${COMP_DIR}/targets.parquet")
fi
if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: data not found under DATA_DIR='$DATA_DIR' (cwd: $PWD). Missing:" >&2
    printf '         %s\n' "${missing[@]}" >&2
    data_dir_diagnosis "$DATA_DIR"
    exit 1
fi

if [ -L "$DATA_DIR" ]; then
    echo "Data under '$DATA_DIR' -> $(readlink "$DATA_DIR"):"
else
    echo "Data under '$DATA_DIR':"
fi
printf '  %-46s %8s   %s\n' file size modified
for f in "${DATA_DIR}/train.parquet" "${DATA_DIR}/test.parquet" \
         "${COMP_DIR}/template.parquet" "${COMP_DIR}/targets.parquet"; do
    [ -f "$f" ] || continue
    sz=$(du -h "$f" | cut -f1)
    mt=$(date -r "$f" '+%F %H:%M' 2>/dev/null || stat -c '%y' "$f" 2>/dev/null | cut -c1-16)
    printf '  %-46s %8s   %s\n' "$f" "$sz" "$mt"
done
echo

# ---- preflight 2: the weights are here -------------------------------------
WEIGHTS="${WEIGHTS:-weights/cgm_jepa_hf}"
ENCODERS="${ENCODERS:-x_cgm_jepa cgm_jepa}"
FILLS="${FILLS:-interp sentinel}"
for e in $ENCODERS; do
    if [ ! -f "${WEIGHTS}/${e}/model.safetensors" ]; then
        echo "ERROR: ${WEIGHTS}/${e}/model.safetensors not found." >&2
        echo "       The weights ride along with ./upload_to_cluster.sh." >&2
        exit 1
    fi
done

# ---- preflight 3: don't append into a tree that already has rows -----------
if [ "${ALLOW_EXISTING:-0}" != "1" ] && [ -d "results/${RESULTS_DATE}" ]; then
    n=$(find "results/${RESULTS_DATE}" -name '*_results.csv' 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ]; then
        echo "ERROR: results/${RESULTS_DATE}/ already holds $n result CSV(s)." >&2
        echo "       These files APPEND and never dedupe, so submitting into this" >&2
        echo "       tree would give you duplicate rows per cell. Pick a fresh" >&2
        echo "       --date, or pass ALLOW_EXISTING=1 if you are deliberately" >&2
        echo "       refilling gaps (with the phase toggles set to match)." >&2
        exit 1
    fi
fi

# Largest first: Loop is 77% of the live template, so it is the long pole and
# should start before the small datasets rather than behind them.
SOURCES="${SOURCES:-Loop ReplaceBG IOBP2 PEDAP BrisT1D CTR3 HUPA-UCM AZT1D ShanghaiT1DM}"
SHARD_BY_SOURCE="${SHARD_BY_SOURCE:-0}"

# shellcheck source=sbatch_overrides.sh
. ./sbatch_overrides.sh
sb_overrides_banner "jepa-all"
# Print the EFFECTIVE values, not the driver header's -- an override banner that
# contradicts the line under it is worse than no line at all.
echo "Effective: account=${ACCOUNT} partition=${PARTITION:-oxf-c64-m512}" \
     "cpus=${CPUS:-8} mem=${MEM:-16G} gpus=${GPUS:-0} time=${WALLTIME:-04:00:00}"
echo "  (values not overridden come from run_jepa_cluster.sh's #SBATCH header)"
echo "Configs: encoders='${ENCODERS}' fills='${FILLS}' competition=${COMPETITION}"
echo "Phases:  sub=${RUN_SUB:-1} full=${RUN_FULL:-1}   shard_by_source=${SHARD_BY_SOURCE}"
echo

# ---- submit ----------------------------------------------------------------
DRY_RUN="${DRY_RUN:-0}"
n_jobs=0

submit() {   # $1 = job name, $2 = extra --export assignments (may be empty)
    local name="$1" extra="$2"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] sbatch --job-name=${name} ${SB_OVERRIDES[*]}" \
             "--export=ALL${extra} run_jepa_cluster.sh --date ${RESULTS_DATE}"
    else
        sbatch --job-name="$name" \
            "${SB_OVERRIDES[@]}" \
            --export=ALL"${extra}" \
            run_jepa_cluster.sh --date "$RESULTS_DATE"
    fi
    n_jobs=$((n_jobs + 1))
}

for e in $ENCODERS; do
    for f in $FILLS; do
        base=",ENCODER=${e},FILL=${f},COMPETITION=${COMPETITION}"
        if [ "$SHARD_BY_SOURCE" != "1" ]; then
            submit "jepa_${e}_${f}" "$base"
            continue
        fi
        first=1
        for s in $SOURCES; do
            # The subsample is a whole-mix metric, so exactly one shard runs it.
            sub=$([ "$first" = "1" ] && echo "${RUN_SUB:-1}" || echo 0)
            first=0
            tag=$(printf '%s' "$s" | tr -cd '[:alnum:]')
            submit "jepa_${e}_${f}_${tag}" "${base},ONLY_SOURCE=${s},RUN_SUB=${sub}"
        done
    done
done

echo
echo "Submitted ${n_jobs} job(s) for results date ${RESULTS_DATE}."
echo
echo "Watch:    squeue -u \$USER -o '%.10i %.24j %.2t %.10M %R'"
echo "Progress: logs/${RESULTS_DATE}/jepa_zeroshot/<encoder>_<fill>_<shard>_<jobid>.out"
echo "Startup:  logs/sbatch/jepa_*_<jobid>.err"
echo "Results:  results/${RESULTS_DATE}/jepa_zeroshot/"
echo
if [ "$SHARD_BY_SOURCE" = "1" ]; then
    echo "Sharded run -- reassemble each configuration before validating:"
    echo "  python assemble_submission.py --date ${RESULTS_DATE} --tag jepa_<encoder>_ridge_<fill>"
fi
echo "Then pull it down:  ./download_from_cluster.sh ${RESULTS_DATE}"
