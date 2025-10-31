#!/bin/bash

#SBATCH --job-name=Ablation5
#SBATCH --output=train_baseline.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=10:00:00

module load mamba
mamba activate staveq2
export TOKENIZERS_PARALLELISM=True

srun --nodes=1 --ntasks=1 --unbuffered python3 train_baseline.py
