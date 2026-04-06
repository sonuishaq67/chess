#!/bin/bash
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 0-16:00:00
#SBATCH -p general
#SBATCH -q private
#SBATCH -G a100:1
#SBATCH --mem=128G
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ishaik4@asu.edu

source /home/ishaik4/miniconda3/etc/profile.d/conda.sh
conda activate chess

export PYTHONUNBUFFERED=1

cd ~/chess
./cli.sh train
