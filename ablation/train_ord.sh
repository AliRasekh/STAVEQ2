#!/bin/bash

#SBATCH --job-name=Ablation4
#SBATCH --output=train_ord.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=10:00:00

module load mamba
mamba activate staveq2
export TOKENIZERS_PARALLELISM=True

# the base model is same as model_0_25, uncomment if not already created
# srun --nodes=1 --ntasks=1 --unbuffered --gpus=0 python3 create_base_0_25.py
srun --nodes=1 --ntasks=1 --unbuffered python3 train_ord.py
