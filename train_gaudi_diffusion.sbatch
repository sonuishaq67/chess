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

# Targets the gaudi-pytorch-diffusion-1.22.0.740 env where
# torch.compile(backend='hpu_backend') actually works for training (verified in
# test_diffusion_env.sbatch / slurm.51612512). The pytorch-2.9.0-gaudi env has a
# broken _recipe_compiler_C.so ABI. Use this sbatch to run the compile-enabled
# path; use train_gaudi_fast.sbatch (+ --no-hpu-graphs) as the lazy-mode fallback.
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
export PT_HPU_RECIPE_CACHE_CONFIG=/tmp/recipe_cache_${SLURM_JOB_ID},true,1024,false
# Bypass the jit-fork recipe compiler path whose fork_pybind/_recipe_compiler_C.so
# is missing habana::graph::GraphStorage::get() on this install. Non-fork path is
# selected by passes.py:1268/2000 via ctx.use_jit_fork = bc.get_pt_hpu_use_jit_fork().
export PT_HPU_USE_JIT_FORK=0

NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
if [[ -n "${HABANA_VISIBLE_MODULES:-}" ]]; then
    IFS=',' read -r -a visible_modules <<< "$HABANA_VISIBLE_MODULES"
    NPROC_PER_NODE="${#visible_modules[@]}"
fi
echo "Launching torchrun (diffusion env: FusedSDPA + FusedAdamW + torch.compile + no_sync) with nproc_per_node=$NPROC_PER_NODE"

cd ~/chess
torchrun --nproc_per_node="$NPROC_PER_NODE" -m src.training.train_fast \
    --config configs/training.yml \
    --model-config configs/model.yml \
    "$@"
