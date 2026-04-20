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
    --bind "$REPO":/workspace/chess
    --bind "$DATA":/workspace/chess/dataset/packing:ro
)
# Apptainer's directory bind of /dev/accel doesn't preserve char-device
# semantics for Synapse (hl-smi works, synDeviceAcquire fails with "No devices
# found"). File-level binds of each device node do work — so enumerate
# /dev/accel/* on the host and bind each one individually.
if [[ -d /dev/accel ]]; then
    for d in /dev/accel/*; do
        [[ -c "$d" ]] && BINDS+=(--bind "$d:$d")
    done
fi
# Apptainer already mounts /sys from host — don't overmount /sys/class/accel.

ENVS=(
    --env PYTHONUNBUFFERED=1
    --env PT_HPU_LAZY_MODE=1
    --env PT_HPU_ENABLE_LAZY_COLLECTIVES=0
    --env "PT_HPU_RECIPE_CACHE_CONFIG=/tmp/recipe_cache_${SLURM_JOB_ID},true,1024,false"
    --env "HABANA_VISIBLE_MODULES=${HABANA_VISIBLE_MODULES:-}"
)

echo "Launching torchrun inside Apptainer with nproc_per_node=$NPROC_PER_NODE"
echo "Binding ${#BINDS[@]} items (including $(ls /dev/accel/ 2>/dev/null | wc -l) accel device nodes)"

# `bash -c` (no -l) — avoids the host modules.sh /dev/null cgroup issue from
# bash -lc, while still letting us source Habana's profile.d so GC_KERNEL_PATH,
# HABANA_PLUGINS_LIB_PATH, HABANA_SCAL_BIN_PATH, etc. are set for Synapse.
apptainer exec --pwd /workspace/chess --writable-tmpfs "${BINDS[@]}" "${ENVS[@]}" "$SIF" \
    bash -c '
        for f in /etc/profile.d/habanalabs*.sh; do [ -f "$f" ] && source "$f"; done
        exec torchrun --nproc_per_node='"$NPROC_PER_NODE"' -m src.training.train_fast \
            --config configs/training.yml \
            --model-config configs/model.yml \
            '"$*"'
    '
