#!/bin/bash

# examples of flags:
    # ./submit_jobs.sh --prompt_verbose --thinking_verbose "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # ./submit_jobs.sh --thinking_verbose "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # ./submit_jobs.sh --start_vllm_verbose "meta-llama/Meta-Llama-3.1-8B-Instruct" 

# Default models if none provided
DEFAULT_MODELS=(
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "meta-llama/Meta-Llama-3.1-70B-Instruct"
    "google/gemma-2-9b-it"
    "google/gemma-2-27b-it"
    "google/gemma-3-4b-it"
    "google/gemma-3-27b-it"
    "Qwen/QwQ-32B"
    "NovaSky-AI/Sky-T1-32B-Preview"
)

# --- Argument Parsing ---
VERBOSE_FLAG=""
THINKING_VERBOSE_FLAG=""
PROMPT_VERBOSE_FLAG=""
MODELS=()

for arg in "$@"; do
  case $arg in
    --start_vllm_verbose)
      VERBOSE_FLAG="--start_vllm_verbose"
      ;;
    --thinking_verbose)
      THINKING_VERBOSE_FLAG="--thinking_verbose"
      ;;
    --prompt_verbose)
      PROMPT_VERBOSE_FLAG="--prompt_verbose"
      ;;
    *)
      # Assume anything else is a model name
      MODELS+=("$arg")
      ;;
  esac
done

# Get the absolute path to the run script
SCRIPT_PATH=$(realpath "$(dirname "$0")/run_hierarchical_reasoning.slurm")

# If no models were provided in the arguments, use the default list
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "No models specified, using default list:"
    printf '%s\n' "${DEFAULT_MODELS[@]}"
    echo ""
    MODELS=("${DEFAULT_MODELS[@]}")
fi

# --- Job Submission ---
for MODEL in "${MODELS[@]}"; do
    echo "Submitting job for model: $MODEL"
    # Export the verbose flags to the slurm script
    sbatch --export=SCRIPT_PATH="$SCRIPT_PATH",MODEL="$MODEL",VLLM_VERBOSE_FLAG="$VERBOSE_FLAG",THINKING_VERBOSE_FLAG="$THINKING_VERBOSE_FLAG",PROMPT_VERBOSE_FLAG="$PROMPT_VERBOSE_FLAG" "$SCRIPT_PATH"
done

echo "All jobs submitted!" 