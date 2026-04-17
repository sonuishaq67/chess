#!/bin/bash -l
#SBATCH -p gaudi
#SBATCH -q class_gaudi
#SBATCH -N 1
#SBATCH -G 8
#SBATCH -t 0-16:00:00
#SBATCH -c 144
#SBATCH --mem=448G
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ishaik4@asu.edu

# Use ASU's prebuilt Gaudi env (habana-torch-plugin matched to torch 2.9.0).
source /home/ishaik4/miniconda3/etc/profile.d/conda.sh
conda activate /packages/envs/pytorch-2.9.0-gaudi

# Sanity: /etc/profile.d/habanalabs.sh (auto-sourced because bash -l) should
# have set GC_KERNEL_PATH. If this prints empty, the shebang got lost.
echo "GC_KERNEL_PATH=$GC_KERNEL_PATH"
echo "Python: $(which python)"
echo "HABANA_VISIBLE_MODULES=${HABANA_VISIBLE_MODULES:-<unset>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"

export PYTHONUNBUFFERED=1
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=0

NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
if [[ -n "${HABANA_VISIBLE_MODULES:-}" ]]; then
    IFS=',' read -r -a visible_modules <<< "$HABANA_VISIBLE_MODULES"
    NPROC_PER_NODE="${#visible_modules[@]}"
fi
echo "Launching torchrun with nproc_per_node=$NPROC_PER_NODE"

cd ~/chess
torchrun --nproc_per_node="$NPROC_PER_NODE" -m src.training.train \
    --config configs/training.yml \
    --model-config configs/model.yml
