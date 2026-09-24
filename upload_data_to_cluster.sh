#!/usr/bin/env bash
# =============================================================================
# Push the local data/ folder up to the cluster over rsync/ssh.
#
# Why a separate script: upload_to_cluster.sh deliberately EXCLUDES data/ (code
# flows up, results and logs flow down, and the cluster keeps its own data
# copy). data/ is ~1.5 GB and changes rarely, so it should not ride along with
# every code edit. This is the explicit path for syncing it -- when the
# MetaboNet export is refreshed, or when setting the cluster up for the first
# time. rsync only transfers changed files, so pushing the whole folder costs
# only what actually changed.
#
# THE DATA LIVES ON SCRATCH, not in the code tree. Home directories are
# quota'd and backed up; 1.5 GB of parquet belongs on the fast, large,
# unbacked-up filesystem instead. This script pushes to GP_REMOTE_DATA_DIR and
# then symlinks <code tree>/data at it, so every driver keeps working with the
# plain DATA_DIR=data default and nothing has to know where the bytes are.
#
# Scratch is usually PURGED on a schedule. If a job fails its fail-fast data
# guard weeks from now, the first thing to check is whether scratch was swept --
# re-running this script restores it.
#
# The four files the experiments need:
#   data/train.parquet                        fitting data for the ridge head
#   data/test.parquet                         model inputs for both leaderboards
#   data/<competition>/template.parquet       the rows to predict, and their order
#   data/live_leaderboard/targets.parquet     ground truth, for the scored subsample
#
# The guard below refuses a push whose template and targets disagree on row
# count. run.py compares them element-wise by position, so a mismatched pair
# does not fail loudly on the cluster -- it scores the wrong rows against each
# other, or dies hours into a sweep. Cheaper to catch here.
#
# Usage:
#   ./upload_data_to_cluster.sh              push for real
#   ./upload_data_to_cluster.sh --dry-run    preview everything, copies nothing
#   ./upload_data_to_cluster.sh --delete     mirror deletions: remote-only files
#                                            under data/ are REMOVED. Any extra
#                                            rsync flag is passed through.
#
# Point this at a different cluster with GP_REMOTE_USER / GP_REMOTE_HOST /
# GP_REMOTE_DIR.
# =============================================================================

set -euo pipefail

# Site configuration. Override via the environment rather than editing here:
#   GP_REMOTE_USER / GP_REMOTE_HOST / GP_REMOTE_DIR
REMOTE_USER="${GP_REMOTE_USER:-vmli3}"
REMOTE_HOST="${GP_REMOTE_HOST:-cirrostratus.it.emory.edu}"
REMOTE_DIR="${GP_REMOTE_DIR:-/users/vmli3/glucose-pred}"
# Where the parquet actually lands. Override for a different scratch layout.
REMOTE_DATA_DIR="${GP_REMOTE_DATA_DIR:-/scratch/${REMOTE_USER}/glucose-pred/data}"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
DATA_DIR="${DATA_DIR:-data}"

if [ ! -d "${LOCAL_DIR}/${DATA_DIR}" ]; then
    echo "ERROR: ${LOCAL_DIR}/${DATA_DIR} does not exist -- nothing to push." >&2
    exit 1
fi

missing=()
for f in train.parquet test.parquet live_leaderboard/template.parquet; do
    [ -f "${LOCAL_DIR}/${DATA_DIR}/${f}" ] || missing+=("${DATA_DIR}/${f}")
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: required files are absent locally:" >&2
    printf '         %s\n' "${missing[@]}" >&2
    echo "       train/test come from metabo-net.org; the templates and targets" >&2
    echo "       are Git LFS objects -- restore them with 'git lfs pull'." >&2
    exit 1
fi

# Template and targets are row-for-row aligned on (id, source_file, date) and
# run.py compares them positionally. A pair that disagrees on length is a
# corrupt or half-pulled LFS object, which is worth catching before it reaches
# shared storage.
TGT="${LOCAL_DIR}/${DATA_DIR}/live_leaderboard/targets.parquet"
if [ -f "$TGT" ]; then
    python3 - "${LOCAL_DIR}/${DATA_DIR}/live_leaderboard/template.parquet" "$TGT" <<'PYEOF' || exit 1
import sys
import pyarrow.parquet as pq

tmpl_path, tgt_path = sys.argv[1:3]
n_tmpl = pq.ParquetFile(tmpl_path).metadata.num_rows
n_tgt = pq.ParquetFile(tgt_path).metadata.num_rows
if n_tmpl != n_tgt:
    sys.exit(f"ERROR: template has {n_tmpl:,} rows but targets has {n_tgt:,}.\n"
             f"       They must be row-for-row aligned. One of the two is a\n"
             f"       stale or partially fetched LFS object -- run 'git lfs pull'\n"
             f"       and check the sizes (~19 MB and ~33 MB) before pushing.")
print(f"  template/targets agree: {n_tmpl:,} rows each")
PYEOF
else
    echo "  note: no targets.parquet locally -- the scored subsample phase will not run."
fi

echo "Pushing:"
du -h "${LOCAL_DIR}/${DATA_DIR}"/*.parquet "${LOCAL_DIR}/${DATA_DIR}"/*/*.parquet 2>/dev/null \
    | sed 's|^|  |' || true
echo

SSH_MUX="-o ControlMaster=auto -o ControlPath=/tmp/gp_sync_cluster-%C -o ControlPersist=60s"

# Create the scratch tree, then point <code tree>/data at it. `ln -sfn` is
# deliberate: -n stops a re-run from nesting the link inside the directory it
# already points at, which is what a bare `ln -sf` does the second time and
# leaves you with data/data. Refuse to clobber a real directory -- if one is
# there, somebody has data in the code tree and deleting it silently is not
# this script's call.
ssh ${SSH_MUX} "${REMOTE}" "
  set -e
  mkdir -p '${REMOTE_DATA_DIR}' '${REMOTE_DIR}'
  if [ -d '${REMOTE_DIR}/${DATA_DIR}' ] && [ ! -L '${REMOTE_DIR}/${DATA_DIR}' ]; then
    echo \"ERROR: ${REMOTE_DIR}/${DATA_DIR} is a real directory, not a symlink.\" >&2
    echo \"       Move or remove it, then re-run. Refusing to replace it.\" >&2
    exit 1
  fi
  ln -sfn '${REMOTE_DATA_DIR}' '${REMOTE_DIR}/${DATA_DIR}'
"

echo "--- push: ${DATA_DIR}/ -> ${REMOTE}:${REMOTE_DATA_DIR}/ ---"
rsync -avzh --progress -e "ssh ${SSH_MUX}" "$@" \
  --exclude='.DS_Store' \
  --exclude='*.py' \
  "${LOCAL_DIR}/${DATA_DIR}/" \
  "${REMOTE}:${REMOTE_DATA_DIR}/"

echo
echo "Pushed ${DATA_DIR}/ -> ${REMOTE}:${REMOTE_DATA_DIR}/"
echo "Linked ${REMOTE_DIR}/${DATA_DIR} -> ${REMOTE_DATA_DIR}"
echo
echo "Verify from the login node:"
echo "  ls -l ${REMOTE_DIR}/${DATA_DIR} && du -sh ${REMOTE_DATA_DIR}"
