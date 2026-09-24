#!/usr/bin/env bash
# =============================================================================
# Push this repository's CODE to the compute cluster over rsync/ssh.
#
# Code moves up; results and logs move down. Experiments are written by Slurm on
# the remote, so pull them back with ./download_from_cluster.sh rather than
# letting rsync mirror in both directions -- a two-way sync over a results tree
# that is being appended to by running jobs is a good way to lose rows.
#
# ALLOW-LIST, NOT A BLOCK-LIST. Everything is refused by the trailing
# --exclude='*'; only the patterns named above it are sent. A block-list fails
# open: every new local directory ships until someone remembers to add it. An
# allow-list fails closed, which is the safe direction when `data/` holds 1.5 GB
# of clinical CGM. If you add a file type the jobs need at runtime, add its
# --include line here, or the cluster runs without it.
#
# WHAT IS SENT: root-level *.py and *.sh, requirements*.txt, the root *.md
# orientation files (README / CLAUDE / EXPERIMENTS, useful when debugging from
# an ssh session), and weights/.
#
# weights/ is the ONE directory that ships. It is 8 MB of published CGM-JEPA
# encoder checkpoints, and compute nodes frequently have no route to the
# Hugging Face hub, so fetching them at job start is not dependable. The
# hub-download cache inside it is excluded -- it is bookkeeping, not weights.
#
# data/ is pushed separately by ./upload_data_to_cluster.sh: it is ~1.5 GB,
# changes rarely, and should not ride along with every code edit.
#
# Under --delete, an excluded path is also PROTECTED on the receiver, so the
# remote results/, logs/ and data/ trees are never touched by this script.
#
# Usage:
#   ./upload_to_cluster.sh              push for real
#   ./upload_to_cluster.sh --dry-run    preview; copies nothing
#   ./upload_to_cluster.sh --delete     mirror deletions on the remote code tree
#
# Any additional rsync flag is passed through. Point this at a different cluster
# with GP_REMOTE_USER / GP_REMOTE_HOST / GP_REMOTE_DIR.
# =============================================================================

set -euo pipefail

# Site configuration. Override via the environment rather than editing here:
#   GP_REMOTE_USER / GP_REMOTE_HOST / GP_REMOTE_DIR
REMOTE_USER="${GP_REMOTE_USER:-vmli3}"
REMOTE_HOST="${GP_REMOTE_HOST:-cirrostratus.it.emory.edu}"
REMOTE_DIR="${GP_REMOTE_DIR:-/users/vmli3/glucose-pred}"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

# One shared, reused ssh connection for everything below (mkdir + rsync), so you
# only enter your passphrase/Duo once per run.
SSH_MUX="-o ControlMaster=auto -o ControlPath=/tmp/gp_sync_cluster-%C -o ControlPersist=60s"

# logs/sbatch/ holds the #SBATCH --output/--error stubs: everything a driver
# emits BEFORE it redirects into the dated folder -- conda init, the fail-fast
# guards, and slurmstepd's own OOM/preemption messages. Slurm does NOT create
# intermediate directories, so a missing logs/sbatch/ fails the job instantly.
# Creating it here covers the direct-`sbatch` path; the fan-out also mkdirs it
# via sbatch_overrides.sh.
ssh ${SSH_MUX} "${REMOTE}" "mkdir -p '${REMOTE_DIR}' '${REMOTE_DIR}/logs/sbatch'"

echo "--- push: code -> ${REMOTE}:${REMOTE_DIR} ---"
# Rules are evaluated in order and the FIRST match wins, so the exclusions below
# are stated before the includes that would otherwise pull them in.
rsync -avzh --progress -e "ssh ${SSH_MUX}" "$@" \
  --exclude='/weights/**/.cache/' \
  --exclude='/weights/**/*.md' \
  --exclude='__pycache__/' \
  --include='/*.py' \
  --include='/*.sh' \
  --include='/*.md' \
  --include='/requirements.txt' \
  --include='/requirements-jepa.txt' \
  --include='/weights/' \
  --include='/weights/**' \
  --exclude='*' \
  "${LOCAL_DIR}/" \
  "${REMOTE}:${REMOTE_DIR}/"

echo
echo "Pushed ${LOCAL_DIR} -> ${REMOTE}:${REMOTE_DIR}"
echo "Next:  ./upload_data_to_cluster.sh    (only when data/ has changed)"
echo "Then:  ssh ${REMOTE} 'cd ${REMOTE_DIR} && bash submit_jepa_all.sh --date \$(date +%F)'"
