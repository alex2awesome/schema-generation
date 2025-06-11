#!/bin/bash

#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --partition=sched_mit_psfc_gpu_r8
#SBATCH --mem=256G
#SBATCH --job-name=hierarchical_reasoning
#SBATCH --output=logs/hierarchical_reasoning_%j.out
#SBATCH --error=logs/hierarchical_reasoning_%j.err

# Load any necessary modules (adjust as needed for your cluster)

source /home/spangher/.bashrc
conda activate /home/spangher/miniconda3/envs/alex_clean

module load cuda/12.4.0

# Verify SCRIPT_PATH is set
if [ -z "$SCRIPT_PATH" ]; then
    echo "Error: SCRIPT_PATH environment variable is not set"
    exit 1
fi

# Verify MODEL is set
if [ -z "$MODEL" ]; then
    echo "Error: MODEL environment variable is not set"
    exit 1
fi

echo "Running with model: $MODEL"
echo "Script path: $SCRIPT_PATH"

# Set up environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SET_PRIORITY=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# Define the base command arguments as an array
BASE_ARGS=(
    "--input_file" "$(dirname "$SCRIPT_PATH")/../../qwq-32b/make_hierarchy/qwq-32b-rollouts-labeled.json"
    "--label_level" "3"
    "--temperature" "0.1"
    "--n_rows" "100"
    "--max_reasoning_steps" "15"
    "--category" "chess_puzzle"
    "--ground_truth_format" "chess coordinate notation"
    "--llm_framework" "vllm"
    "--vllm_base_url" "http://localhost:8000/v1"
    "--vllm_gpus" "0,1,2,3"
    "--vllm_auto_start"
    "--tree_model_path" "$(dirname "$SCRIPT_PATH")/../../qwq-32b/models/chess_agglomerative_clustering_outputs__discretized__labels_descriptions__child-nodes__output-labels-desc"
)

# Create a unique log file for this model
MODEL_SAFE=$(echo $MODEL | tr '/' '_' | tr '-' '_')
LOG_FILE="$(dirname "$SCRIPT_PATH")/logs/hierarchical_reasoning_${MODEL_SAFE}_$(date +%Y%m%d_%H%M%S).log"

# Run the hierarchical reasoning script
# Verbose flags are passed from submit_jobs.sh and will be either empty or the flag name
python "$(dirname "$SCRIPT_PATH")/../hierarchical_reasoning.py" \
    --model_name "$MODEL" \
    "${BASE_ARGS[@]}" \
    $VLLM_VERBOSE_FLAG \
    $THINKING_VERBOSE_FLAG \
    $PROMPT_VERBOSE_FLAG \
    2>&1 | tee "$LOG_FILE"

# Check if the command succeeded
if [ $? -eq 0 ]; then
    echo "Successfully completed experiment with $MODEL"
else
    echo "Error occurred with model $MODEL"
    exit 1
fi

echo "Completed: $MODEL at $(date)" 