#!/bin/bash

#SBATCH --job-name=Internvideo2Training
#SBATCH --output=train.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --gpus=2
#SBATCH --time=14:00:00

source /opt/conda/etc/profile.d/conda.sh
conda activate internvideo2

srun --nodes=1 --ntasks=1 accelerate launch --num_processes=2 --main_process_port=37200 train_stacked_lora.py
