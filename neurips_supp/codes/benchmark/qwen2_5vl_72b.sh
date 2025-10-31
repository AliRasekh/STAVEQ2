#!/bin/bash

#SBATCH --job-name=Q25-72B
#SBATCH --output=qwen2_5vl_72b.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gpus=4
#SBATCH --time=24:00:00

module load mamba
mamba activate benchmark2

export HF_HOME=$HOME/.cache/huggingface
export HF_TOKEN=

srun --nodes=1 --ntasks=1 --unbuffered \
    python3 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-72B-Instruct,device_map=auto,use_flash_attention_2=True,max_pixels=200704 \
    --tasks vitatecs \
    --batch_size 1 \
    --log_samples --log_samples_suffix reproduce --output_path ./logs
