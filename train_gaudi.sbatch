#!/bin/bash -l
#SBATCH -p gaudi
#SBATCH -q class_gaudi
#SBATCH -N 1
#SBATCH -G 8
#SBATCH -t 0-16:00:00
#SBATCH -c 144
#SBATCH --mem=128G
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

export PYTHONUNBUFFERED=1
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=1

cd ~/chess
torchrun --nproc_per_node=8 -m src.training.train \
    --config configs/training.yml \
    --model-config configs/model.yml
