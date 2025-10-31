#!/bin/bash

#SBATCH --job-name=Q2-7B
#SBATCH --output=qwen2vl_7b.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=2
#SBATCH --time=24:00:00

module load mamba
mamba activate benchmark

export HF_HOME=$HOME/.cache/huggingface
export HF_TOKEN=

srun --nodes=1 --ntasks=1 --unbuffered \
    accelerate launch --main_process_port 5881 --num_processes=2 -m lmms_eval \
    --model qwen2_vl \
    --model_args=pretrained=Qwen/Qwen2-VL-7B-Instruct,max_pixels=200704 \
    --tasks vitatecs \
    --batch_size 1 \
    --log_samples --log_samples_suffix reproduce --output_path ./logs
