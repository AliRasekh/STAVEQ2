#!/bin/bash

#SBATCH --job-name=STAVEQ2_Training
#SBATCH --output=train.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gpus=8
#SBATCH --time=96:00:00

module load mamba
mamba activate staveq2
export TQDM_DISABLE=1
export TOKENIZERS_PARALLELISM=True 

# run this step only once to create the base model with random temporal attn weights
# comment this when resuming the training
srun --nodes=1 --ntasks=1 --unbuffered --gpus=0 python3 create_base.py

srun --nodes=1 --ntasks=1 --unbuffered accelerate launch --main_process_port 8570 --num_processes 8 train.py
