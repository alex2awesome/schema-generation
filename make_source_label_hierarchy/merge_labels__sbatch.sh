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

python merge_labels.py \
    --input_data_file ../../data/nodes_to_cluster.csv \
    --input_col_name col_to_cluster \
    --trained_sbert_model_name models/mpnet-base-all-nli-triplet/trained-model \
    --output_cluster_file ../../data/preliminary_clusters.npy \
    --output_data_file ../../data/nodes_with_preliminary_clusters.csv
