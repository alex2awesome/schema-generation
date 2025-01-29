#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=50
#SBATCH --mem=150G
#SBATCH --partition=sched_mit_psfc_gpu_r8

# Activate your environment or load modules if needed
source /home/spangher/.bashrc
conda activate alex

# Run the script with desired parameters
python train_sentence_similarity_model.py \
    --model_name 'microsoft/mpnet-base' \
    --data_file_pattern '../../data/triplets_*.jsonl' \
    --output_dir 'models/mpnet-base-all-nli-triplet' \
    --num_train_epochs 3 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --fp16 \
    --eval_strategy 'steps' \
    --eval_steps 20000 \
    --save_strategy 'epoch' \
    --save_total_limit 2 \
    --logging_steps 10000 \
    --run_name 'mpnet-base-all-nli-triplet' \
    --test_size 0.1 \
    --do_initial_evaluation
