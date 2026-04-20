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

# Targets the gaudi-pytorch-diffusion-1.22.0.740 env but now uses Habana's
# training HPU graph wrapper (ModuleCacher) in lazy mode instead of
# torch.compile. The earlier compile validation for this env was done on a toy
# model and was too weak to prove the full Transformer+DDP training path.
source /home/ishaik4/miniconda3/etc/profile.d/conda.sh
conda activate /packages/envs/gaudi-pytorch-diffusion-1.22.0.740

echo "GC_KERNEL_PATH=$GC_KERNEL_PATH"
echo "Python: $(which python)"
echo "torch: $(python -c 'import torch; print(torch.__version__)')"
echo "HABANA_VISIBLE_MODULES=${HABANA_VISIBLE_MODULES:-<unset>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"

export PYTHONUNBUFFERED=1
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=0
export PT_HPU_RECIPE_CACHE_CONFIG=/scratch/$USER/gaudi_recipe_cache,true,4096,false

NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
if [[ -n "${HABANA_VISIBLE_MODULES:-}" ]]; then
    IFS=',' read -r -a visible_modules <<< "$HABANA_VISIBLE_MODULES"
    NPROC_PER_NODE="${#visible_modules[@]}"
fi
echo "Launching torchrun (diffusion env: FusedSDPA + FusedAdamW + ModuleCacher HPU graphs + no_sync) with nproc_per_node=$NPROC_PER_NODE"

cd ~/chess
torchrun --nproc_per_node="$NPROC_PER_NODE" -m src.training.train_fast \
    --config configs/training.yml \
    --model-config configs/model.yml \
    --graph-mode modulecacher \
    --graph-max-graphs 1 \
    "$@"
