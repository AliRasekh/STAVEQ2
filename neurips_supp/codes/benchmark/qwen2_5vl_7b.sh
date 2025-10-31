#!/bin/bash

#SBATCH --job-name=Q25-7B
#SBATCH --output=qwen2_5vl_7b.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=2
#SBATCH --time=24:00:00

module load mamba
mamba activate benchmark2

export HF_HOME=$HOME/.cache/huggingface
export HF_TOKEN=

srun --nodes=1 --ntasks=1 --unbuffered \
    accelerate launch --main_process_port 8591 --num_processes=2 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,use_flash_attention_2=True,max_pixels=200704 \
    --tasks vitatecs \
    --batch_size 1 \
    --log_samples --log_samples_suffix reproduce --output_path ./logs
