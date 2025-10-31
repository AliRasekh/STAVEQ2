#!/bin/bash

#SBATCH --job-name=STAVEQ2
#SBATCH --output=staveq2.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gpus=4
#SBATCH --time=24:00:00

module load mamba
mamba activate benchmark

export HF_HOME=$HOME/.cache/huggingface
export HF_TOKEN=
export MODEL_PATH=../staveq2-training/saves/final

srun --nodes=1 --ntasks=1 --unbuffered \
    python3 -m lmms_eval \
    --model custom_qwen2_vl \
    --model_args=pretrained=$MODEL_PATH,device_map=auto,max_pixels=200704 \
    --tasks vitatecs,mvbench,videomme,videomme_w_subtitle \
    --batch_size 1 \
    --log_samples --log_samples_suffix reproduce --output_path ./logs