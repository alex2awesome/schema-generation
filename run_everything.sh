#!/bin/bash

# Set up key variables
EXPERIMENT_DIR="experiments/editorial"
INPUT_FILE="../data/chunks-with-problems.json.gz"
USE_OPENAI=true
MODEL_NAME="gpt-4o-mini"
BATCH_SIZE=500
NUM_SENTS_PER_PROMPT=8
NUM_CENTROIDS=512

# or "meta-llama/Meta-Llama-3.1-70B-Instruct" for VLLM

# Create experiment directory if it doesn't exist
mkdir -p $EXPERIMENT_DIR

# Step 1: Run chunking
# echo "Step 1: Running chunking..."
# python src/run_chunking_prompt.py \
#     --input_data_file $INPUT_FILE \
#     --output_file $EXPERIMENT_DIR/editorial-discourse-input-data.csv

# Step 1: Run initial labeling
echo "Step 1: Running initial labeling..."
python src/step_1__run_initial_labeling_prompts.py \
    --model $MODEL_NAME \
    --input_data_file $EXPERIMENT_DIR/editorial-discourse-input-data.csv \
    --output_file $EXPERIMENT_DIR/editorial-discourse-initial-labeling.json \
    --experiment editorials \
    --start_idx 0 \
    --end_idx 100 \
    --batch_size $BATCH_SIZE \
    --num_sents_per_prompt $NUM_SENTS_PER_PROMPT \
    $([ "$USE_OPENAI" = true ] && echo "--use_openai")

# Step 2: Create supervised similarity data
echo "Step 2: Creating supervised similarity data..."
python src/step_2__create_supervised_similarity_data.py \
    --input_file $EXPERIMENT_DIR/editorial-discourse-initial-labeling-labeling__experiment-editorials__model_${MODEL_NAME//\//-}__0_1635.json \
    --output_file $EXPERIMENT_DIR/${MODEL_NAME//\//-}-similarity-data.jsonl \
    --model_name $MODEL_NAME \
    --use_openai \
    --batch_size $BATCH_SIZE \
    --text_col_name label \
    --text_col_name_2 description \
    --sample_size 500000

# Step 3: Train sentence similarity model
echo "Step 3: Training sentence similarity model..."
python src/step_3__train_sentence_similarity_model.py \
    --model_name "sentence-transformers/all-MiniLM-L6-v2" \
    --data_file $EXPERIMENT_DIR/triplets_${MODEL_NAME//\//-}-similarity-data.jsonl \
    --output_dir $EXPERIMENT_DIR/models/editorial-sentence-similarity-model \
    --num_train_epochs 3 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --run_name "editorial-sentence-similarity-model" \
    --do_initial_evaluation

# Step 4: Run clustering
echo "Step 4: Running clustering..."
python src/step_4__merge_labels.py \
    --input_data_file $EXPERIMENT_DIR/editorial-discourse-initial-labeling-labeling__experiment-editorials__model_${MODEL_NAME//\//-}__0_1635.json \
    --input_col_name label \
    --trained_sbert_model_name $EXPERIMENT_DIR/models/editorial-sentence-similarity-model/trained-model \
    --output_cluster_file $EXPERIMENT_DIR/models/cluster_centroids.npy \
    --output_data_file $EXPERIMENT_DIR/models/all_extracted_discourse_with_clusters.csv \
    --skip_umap \
    --ncentroids $NUM_CENTROIDS

# Step 5: Label clusters
echo "Step 5: Labeling clusters..."
python src/step_5__label_low_level_kmeans_clusters.py \
    --input_file $EXPERIMENT_DIR/models/all_extracted_discourse_with_clusters.csv \
    --output_file $EXPERIMENT_DIR/models/cluster_labels.csv \
    --model $MODEL_NAME \
    --cluster_col cluster \
    --label_superset_col label \
    --n_samples_per_cluster 10 \
    $([ "$USE_OPENAI" = true ] && echo "--use_openai")

# Step 6: Run agglomerative clustering
echo "Step 6: Running agglomerative clustering..."
mkdir -p $EXPERIMENT_DIR/models/agglomerative_clustering_outputs
python src/step_6__agglomerative_clustering.py \
    $EXPERIMENT_DIR/models/cluster_centroids.npy \
    $EXPERIMENT_DIR/models/agglomerative_clustering_outputs \
    --min_clusters 2 \
    --max_clusters 15 \
    --min_cluster_size 2 \
    --method ward \
    --metric euclidean \
    --label_tree \
    --initial_labels $EXPERIMENT_DIR/models/cluster_labels.csv \
    --min_descendants 2 \
    --max_depth 8 \
    --num_samples 10 \
    --examples_df $EXPERIMENT_DIR/models/all_extracted_discourse_with_clusters.csv \
    --num_examples_per_node 10 \
    --examples_cluster_col cluster \
    --examples_sentence_col sentences
    # --no_visualize # Uncomment to skip visualization

echo "All steps completed!" 