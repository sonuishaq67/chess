#!/bin/bash -l
#SBATCH -p gaudi
#SBATCH -q class_gaudi
#SBATCH -N 1
#SBATCH -G 8
#SBATCH -t 3-00:00:00
#SBATCH -c 144
#SBATCH --mem=448G
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ishaik4@asu.edu

# Container-based Gaudi training. Replaces bare-metal conda env which hits
# `_recipe_compiler_C.so: undefined symbol: _ZN6habana5graph12GraphStorage3getEv`
# due to host userspace ABI mismatch. All Habana userspace comes from the SIF;
# host provides only the kernel driver and /dev/accel* nodes.

SIF=/scratch/$USER/sif/chess-gaudi.sif
REPO=$HOME/chess
DATA=/scratch/$USER/chess-data/packing    # TODO: set exact scratch subpath before first run

if [[ ! -f "$SIF" ]]; then
    echo "SIF not found at $SIF — build it first on a login node:" >&2
    echo "  apptainer build $SIF $REPO/chess-gaudi.def" >&2
    exit 1
fi

echo "SIF=$SIF"
echo "HABANA_VISIBLE_MODULES=${HABANA_VISIBLE_MODULES:-<unset>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"

NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
if [[ -n "${HABANA_VISIBLE_MODULES:-}" ]]; then
    IFS=',' read -r -a visible_modules <<< "$HABANA_VISIBLE_MODULES"
    NPROC_PER_NODE="${#visible_modules[@]}"
fi

BINDS=(
    --bind /dev/accel:/dev/accel
    --bind "$REPO":/workspace/chess
    --bind "$DATA":/workspace/chess/dataset/packing:ro
)
# Binding /dev wholesale breaks /dev/null + /dev/urandom under SLURM cgroup.
# Only bind host sysfs/config paths that exist on this node.
for p in /sys/class/accel /sys/class/habanalabs /etc/habanalabs; do
    if [[ -e "$p" ]]; then
        BINDS+=(--bind "$p:$p:ro")
    fi
done

ENVS=(
    --env PYTHONUNBUFFERED=1
    --env PT_HPU_LAZY_MODE=1
    --env PT_HPU_ENABLE_LAZY_COLLECTIVES=0
    --env "PT_HPU_RECIPE_CACHE_CONFIG=/tmp/recipe_cache_${SLURM_JOB_ID},true,1024,false"
    --env "HABANA_VISIBLE_MODULES=${HABANA_VISIBLE_MODULES:-}"
)

echo "Launching torchrun inside Apptainer with nproc_per_node=$NPROC_PER_NODE"

# Call torchrun directly — no `bash -lc`. The login shell sources host
# /etc/profile.d/modules.sh + ~/.bashrc, which redirect to /dev/null under the
# SLURM cgroup view and fail with "Permission denied".
apptainer exec --pwd /workspace/chess --writable-tmpfs "${BINDS[@]}" "${ENVS[@]}" "$SIF" \
    torchrun --nproc_per_node=$NPROC_PER_NODE -m src.training.train_fast \
        --config configs/training.yml \
        --model-config configs/model.yml \
        "$@"
