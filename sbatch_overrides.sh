# =============================================================================
# sbatch_overrides.sh -- shared sbatch CLI override surface for the fan-outs
# =============================================================================
# SOURCED, not executed. submit_jepa_all.sh sources this and passes
# "${SB_OVERRIDES[@]}" to sbatch. sbatch command-line options BEAT the in-script
# #SBATCH directives, so this is how a fan-out retargets the driver without
# editing the driver's header.
#
# Knobs (all env, all optional except ACCOUNT which defaults):
#   ACCOUNT    charging account.  DEFAULT general.
#   QOS        --qos
#   PARTITION  --partition
#   CPUS       --cpus-per-task
#   MEM        --mem
#   GPUS       --gpus
#   WALLTIME   --time
# Anything left unset is simply not emitted, so the driver's own #SBATCH value
# stands. A caller may pre-set its own defaults before sourcing and they are
# respected.
#
# ON GPUS. The encoder is 522 k parameters, so a GPU is not needed for the work
# to finish -- but it is often the faster way to finish, because queue time
# dominates: an idle GPU node beats a full CPU partition regardless of how
# little of the card the job uses. The driver header requests none, so a GPU is
# always an explicit choice:
#   GPUS=1 PARTITION=rp6b-1-gm96-c8-m64 MEM=48G bash submit_jepa_all.sh --date <D>
# predict_jepa.py picks the device up automatically (--device auto) and verifies
# the GPU reproduces the CPU forward pass before using it.
#
# ACCOUNT must match the PARTITION -- not every account is associated with every
# partition, and a mismatch is rejected at submit time with "Invalid account or
# account/partition combination specified". Check what you may use with
#   sacctmgr -nP show assoc user=$USER format=account,partition
# =============================================================================

# SBATCH STUB DIR. The driver's #SBATCH --output/--error point at
# logs/sbatch/%x_%j.{out,err}. Slurm does NOT create intermediate directories --
# a missing logs/sbatch/ makes the job fail instantly with "Unable to open file"
# -- so every submit path must guarantee it exists. This is one of them;
# upload_to_cluster.sh mkdirs it remotely for the direct-`sbatch` path.
mkdir -p logs/sbatch

ACCOUNT="${ACCOUNT:-general}"

SB_OVERRIDES=(--account "$ACCOUNT")
if [ -n "${QOS:-}" ];       then SB_OVERRIDES+=(--qos "$QOS"); fi
if [ -n "${PARTITION:-}" ]; then SB_OVERRIDES+=(--partition "$PARTITION"); fi
if [ -n "${CPUS:-}" ];      then SB_OVERRIDES+=(--cpus-per-task "$CPUS"); fi
if [ -n "${MEM:-}" ];       then SB_OVERRIDES+=(--mem "$MEM"); fi
if [ -n "${GPUS:-}" ];      then SB_OVERRIDES+=(--gpus "$GPUS"); fi
if [ -n "${WALLTIME:-}" ];  then SB_OVERRIDES+=(--time "$WALLTIME"); fi

sb_overrides_banner() {   # $1 = tag for the log line
    echo "[${1:-fan-out}] account=$ACCOUNT${QOS:+ qos=$QOS}${PARTITION:+ partition=$PARTITION}" \
         "${CPUS:+cpus=$CPUS}${MEM:+ mem=$MEM}${GPUS:+ gpus=$GPUS}${WALLTIME:+ time=$WALLTIME}"
}
