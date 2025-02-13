#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=50
#SBATCH --mem=150G
#SBATCH --partition=sched_mit_psfc_gpu_r8

# Activate your environment or load modules if needed
source /home/spangher/.bashrc
conda activate alex

schema_size=$1
python generic_run_prompts.py \
    --input_data_file final_prompt_df__schema_size_${schema_size}.csv.gz \
    --output_file output_data/final_prompt_df__schema_size_${schema_size}_labeled.csv \
    --prompt_col prompt \
    --id_col key
