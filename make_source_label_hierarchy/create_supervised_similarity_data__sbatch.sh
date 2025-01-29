#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=100
#SBATCH --mem=400G
#SBATCH --partition=sched_mit_psfc_gpu_r8

source /home/spangher/.bashrc
conda activate alex

run_idx=$1

python batch_label_paired_source_data.py \
    --input_file ../../data/nodes_to_cluster.csv \
    --output_file ../../data/${run_idx}.jsonl \
    --col_name response \
    --embedding_model_name all-MiniLM-L6-v2 \
    --min_sim_threshold 0.3 \
    --max_sim_threshold 0.9 \
    --sample_size 1000000 \
    --k 5 \
    --batch_size 4000
