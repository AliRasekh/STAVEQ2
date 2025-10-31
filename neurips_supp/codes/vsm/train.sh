#!/bin/bash

#SBATCH --job-name=QwenTrainingVSM
#SBATCH --output=train_base_2B.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --gpus=4
#SBATCH --time=9:00:00


module load mamba
mamba activate qwen2

srun --nodes=1 --ntasks=1 accelerate launch --num_processes=4 --main_process_port=37101 train_base.py
