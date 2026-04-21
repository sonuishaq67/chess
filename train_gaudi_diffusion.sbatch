#!/bin/bash -l
#SBATCH -p gaudi
#SBATCH -q class_gaudi
#SBATCH -N 1
#SBATCH -G 8
#SBATCH -t 1-00:00:00
#SBATCH -c 144
#SBATCH --mem=448G
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ishaik4@asu.edu

# Targets the gaudi-pytorch-diffusion-1.22.0.740 env. Training HPU graphs via
# ModuleCacher wrapping the bare model pre-DDP, with lazy collectives enabled
# so HCCL all-reduce is captured inside the recipe instead of stalling the
# replay on eager collectives.
source /home/ishaik4/miniconda3/etc/profile.d/conda.sh
conda activate /packages/envs/gaudi-pytorch-diffusion-1.22.0.740

echo "GC_KERNEL_PATH=$GC_KERNEL_PATH"
echo "Python: $(which python)"
echo "torch: $(python -c 'import torch; print(torch.__version__)')"
echo "HABANA_VISIBLE_MODULES=${HABANA_VISIBLE_MODULES:-<unset>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"

export PYTHONUNBUFFERED=1
export PT_HPU_LAZY_MODE=1
# Lazy collectives are required when training HPU graphs are in use so HCCL
# all-reduce can be captured inside the cached recipe and overlap with compute;
# with eager collectives (0) the last-accum step stalls outside the graph.
export PT_HPU_ENABLE_LAZY_COLLECTIVES=1
export PT_HPU_RECIPE_CACHE_CONFIG=/scratch/$USER/gaudi_recipe_cache,true,4096,false

NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
if [[ -n "${HABANA_VISIBLE_MODULES:-}" ]]; then
    IFS=',' read -r -a visible_modules <<< "$HABANA_VISIBLE_MODULES"
    NPROC_PER_NODE="${#visible_modules[@]}"
fi
echo "Launching torchrun (diffusion env: FusedSDPA + FusedAdamW + ModuleCacher HPU graphs + lazy collectives + no_sync) with nproc_per_node=$NPROC_PER_NODE"

cd ~/chess
torchrun --nproc_per_node="$NPROC_PER_NODE" -m src.training.train_fast \
    --config configs/training.yml \
    --model-config configs/model.yml \
    --graph-mode modulecacher \
    --graph-max-graphs 1 \
    "$@"
