#!/usr/bin/env bash
# =============================================================================
# Pull RESULTS and LOGS for ONE date back down from the cluster via rsync/ssh.
#
# These are written by Slurm on the remote, so they flow the opposite way from
# code; push code up with ./upload_to_cluster.sh. Both trees are
# date-partitioned (results/YYYY-MM-DD/..., logs/YYYY-MM-DD/...), so you pass
# the date you want and only that day's tree comes down.
#
# EXCLUDED: `.<name>.csv.lock` sidecars. predict_jepa.append_results takes an
# exclusive flock on one of these next to every results CSV so that concurrent
# per-configuration jobs can append safely. They are always ZERO BYTES -- the
# lock lives in the inode, not the contents -- and are pure server-side
# coordination state with no analytical value. If any have already been pulled
# down, deleting them locally is safe:
#   find results logs -name '.*.lock' -delete
#
# The predictions parquets ARE pulled: they are the submission artifacts. A full
# live-leaderboard pass is roughly 85 MB per configuration, so pass
# --exclude='preds/' when you only want the metrics.
#
# Usage:
#   ./download_from_cluster.sh YYYY-MM-DD                  pull that date for real
#   ./download_from_cluster.sh YYYY-MM-DD --dry-run        preview, copies nothing
#   ./download_from_cluster.sh YYYY-MM-DD --exclude='preds/'   metrics and logs only
#   ./download_from_cluster.sh YYYY-MM-DD --delete         mirror deletions -- a
#                                     # remote results/<date>/ with fewer files
#                                     # WILL delete local-only files under that
#                                     # date. Any extra rsync flag after the date
#                                     # is passed through the same way.
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

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

# First positional arg is the date to pull; everything after it is passed
# through to rsync (--dry-run, --delete, --exclude, ...).
DATE="${1:-}"
if [[ ! "${DATE}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "usage: $(basename "$0") YYYY-MM-DD [extra rsync flags]" >&2
    echo "  e.g. $(basename "$0") $(date +%F) --dry-run" >&2
    exit 1
fi
shift

# One shared, reused ssh connection for both pulls, so you only enter your
# passphrase/Duo once per run.
SSH_MUX="-o ControlMaster=auto -o ControlPath=/tmp/gp_sync_cluster-%C -o ControlPersist=60s"

for sub in logs results; do
    echo "--- pull: ${REMOTE}:${REMOTE_DIR}/${sub}/${DATE}/ -> ${LOCAL_DIR}/${sub}/${DATE}/ ---"
    mkdir -p "${LOCAL_DIR}/${sub}/${DATE}"
    rsync -avzh --progress -e "ssh ${SSH_MUX}" \
      --exclude='.*.lock' \
      "$@" \
      "${REMOTE}:${REMOTE_DIR}/${sub}/${DATE}/" \
      "${LOCAL_DIR}/${sub}/${DATE}/"
done

echo
echo "Pulled ${DATE} logs/ + results/ -> ${LOCAL_DIR}"
echo

RES="${LOCAL_DIR}/results/${DATE}/jepa_zeroshot"
if [ -d "$RES" ]; then
    echo "Result rows:"
    for f in "$RES"/*_results.csv; do
        [ -e "$f" ] || continue
        printf '  %-58s %s rows\n' "$(basename "$f")" "$(( $(wc -l < "$f") - 1 ))"
    done
    echo
    echo "Any FAILED rows (a logged failure beats a missing one -- a gap is"
    echo "indistinguishable from a job that never started):"
    if grep -lq ',FAILED,' "$RES"/*_results.csv 2>/dev/null; then
        grep -h ',FAILED,' "$RES"/*_results.csv | cut -d, -f3,4 | sed 's|^|  |'
    else
        echo "  none"
    fi
fi
