#!/bin/bash

# Set up key variables
EXPERIMENT_NAME="hate-speech"
EXPERIMENT_DIR="experiments/$EXPERIMENT_NAME"
INPUT_FILE="data/finegrained_reply_cleaned.csv"
USE_OPENAI=true
MODEL_NAME="gpt-4o-mini"
SUMMARIZER_MODEL_NAME="gpt-4o"
BATCH_SIZE=500
NUM_SENTS_PER_PROMPT=8
NUM_CENTROIDS=128

# or "meta-llama/Meta-Llama-3.1-70B-Instruct" for VLLM

# Create experiment directory if it doesn't exist
mkdir -p $EXPERIMENT_DIR

# Define base output file for step 1
# Step 1: Run initial labeling
STEP1_BASE_OUTPUT="$EXPERIMENT_DIR/$EXPERIMENT_NAME-initial-labeling.json"
echo "Step 1: Running initial labeling..."
python src/step_1__run_initial_labeling_prompts.py \
    --model $MODEL_NAME \
    --input_data_file $EXPERIMENT_DIR/$INPUT_FILE \
    --output_file $STEP1_BASE_OUTPUT \
    --experiment $EXPERIMENT_NAME \
    --batch_size $BATCH_SIZE \
    --num_sents_per_prompt $NUM_SENTS_PER_PROMPT \
    $([ "$USE_OPENAI" = true ] && echo "--use_openai")


# Get the number of lines in the prompt cache file to determine default end_idx
PROMPT_CACHE_FILE="${STEP1_BASE_OUTPUT}-prompt-cache.json"
if [ -f "$PROMPT_CACHE_FILE" ]; then
    DEFAULT_END_IDX=$(wc -l < "$PROMPT_CACHE_FILE")
    echo "Found prompt cache file with $DEFAULT_END_IDX lines"
fi
STEP1_OUTPUT_FILE="$EXPERIMENT_DIR/$EXPERIMENT_NAME-initial-labeling-labeling__experiment-${EXPERIMENT_NAME}__model_${MODEL_NAME//\//-}__0_${DEFAULT_END_IDX}.json"
if [ ! -f "$STEP1_OUTPUT_FILE" ]; then
    STEP1_OUTPUT_FILE=$(ls $EXPERIMENT_DIR/${EXPERIMENT_NAME}-initial-labeling-labeling__experiment-${EXPERIMENT_NAME}__model_${MODEL_NAME//\//-}__*_*.json | sort -V | tail -n 1)
fi

# Step 2: Create supervised similarity data
echo "Step 1 output file: $STEP1_OUTPUT_FILE"
echo "Step 2: Creating supervised similarity data..."
python src/step_2__create_supervised_similarity_data.py \
    --input_file "$STEP1_OUTPUT_FILE" \
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
    --output_dir $EXPERIMENT_DIR/models/$EXPERIMENT_NAME-sentence-similarity-model \
    --num_train_epochs 3 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --run_name "$EXPERIMENT_NAME-sentence-similarity-model" \
    --do_initial_evaluation

# Step 4: Run clustering
echo "Step 4: Running clustering..."
python src/step_4__merge_labels.py \
    --input_data_file "$STEP1_OUTPUT_FILE" \
    --input_col_name label \
    --trained_sbert_model_name $EXPERIMENT_DIR/models/$EXPERIMENT_NAME-sentence-similarity-model/trained-model \
    --output_cluster_file $EXPERIMENT_DIR/models/cluster_centroids.npy \
    --output_data_file $EXPERIMENT_DIR/models/all_extracted_discourse_with_clusters.csv \
    --skip_umap \
    --ncentroids $NUM_CENTROIDS \
    --kmeans_downsample_to 20000

# Step 5: Label clusters
echo "Step 5: Labeling clusters..."
python src/step_5__label_low_level_kmeans_clusters.py \
    --input_file $EXPERIMENT_DIR/models/all_extracted_discourse_with_clusters.csv \
    --output_file $EXPERIMENT_DIR/models/cluster_labels.csv \
    --model $SUMMARIZER_MODEL_NAME \
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
    --initial_cluster_labels_path $EXPERIMENT_DIR/models/cluster_labels.csv \
    --min_descendants 2 \
    --max_depth 8 \
    --num_samples_per_node_for_labeling 10 \
    --experiment $EXPERIMENT_NAME \
    --raw_data_examples_df_path $EXPERIMENT_DIR/data/finegrained_reply_cleaned.csv \
    --datapoint_level_labels_path $EXPERIMENT_DIR/models/all_extracted_discourse_with_clusters.csv \
    --num_examples_per_node 10 \
    --examples_cluster_col cluster \
    --examples_sentence_col sentences \
    --labeling_model $SUMMARIZER_MODEL_NAME \
    --checking_model $MODEL_NAME \
    --use_discretization \
    --discretization_goal "determine the common hate-speech theme in the text" \
    --initial_datapoint_level_label_embeddings "${EXPERIMENT_DIR}/initial_label_embeddings_cache.npy" \
    --batch_size 30 \
    --num_summary_candidates_to_generate 5 \
    --num_datapoints_to_use_for_scoring 200 \
    --num_leaf_samples_to_use_for_generation 30 \
    --use_labels_and_descriptions \
    --use_descriptions_and_examples \
    --use_child_nodes \
    --output_labels_and_descriptions
    # --no_visualize # Uncomment to skip visualization

echo "All steps completed!" 