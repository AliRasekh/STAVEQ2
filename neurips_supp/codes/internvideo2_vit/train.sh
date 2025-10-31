#!/bin/bash

#SBATCH --job-name=Internvideo2Training
#SBATCH --output=runs/train.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --gpus=4
#SBATCH --time=150:00:00

source /opt/conda/etc/profile.d/conda.sh
conda activate internvideo

srun --nodes=1 --ntasks=1  accelerate launch --num_processes=4 --main_process_port=37201 train.py
