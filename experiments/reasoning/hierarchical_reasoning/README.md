# Hierarchical Reasoning Script

This script contains two ways of generating reasoning threads that have a hierarchical structure, and they're centered on two files:

* `hierarchical_reasoning.py` is the main script that contains the hierarchical reasoning logic.
* `train_model_grpo.py` is the script that contains the GRPO training logic to tune the hierarchical decision-making.


## File Structure

```
experiments/reasoning/hierarchical_reasoning/
├── hierarchical_reasoning.py           # Main script for reasoning with pretrained models.
├── train_model_grpo.py                 # Main script to tune base models with GRPO to induce hierarchical reasoning.
├── llm_interface.py                    # Interface for LLM communication -- currently supports OpenAI, VLLM, and TogetherAI. If you wish to add a new LLM framework, you can do so by adding a new class to this file.
├── reasoning_prompts.py                # Stores and manages reasoning prompts
├── README.md                           # This file
├── requirements.txt                    # Requirements file
├── runner_scripts/                     # Scripts for running experiments
│   ├── run_examples.sh
│   ├── run_hierarchical_reasoning.slurm
│   └── submit_jobs.sh
├── util_vllm/                          # Utility scripts for VLLM server management
│   ├── check_vllm_server.py
│   ├── cleanup_vllm_servers.py
│   └── start_vllm_server.py
└── cache/                              # Default cache directory for results and examples
    ├── examples_cache_level_*.pkl
    └── results_*.json
``` 

Dependencies in other parts of the file system:

* `experiments/reasoning/qwq-32b/` contains the QwQ-32B model and the hierarchical clustering outputs.
* `experiments/reasoning/qwq-32b/make_hierarchy/qwq-32b-rollouts-labeled.json` contains the rollouts labeled with the hierarchical clustering outputs.
* `src/utils_openai_client.py` contains the utility scripts for OpenAI client management.
* `src/utils_trees.py` contains the utility scripts for reading the taxonomy tree.


## Example Flow of a Hierarchical Reasoning Experiment

```bash
tree_path = "experiments/reasoning/qwq-32b/models/chess_agglomerative_clustering_outputs__discretized__labels_descriptions__child-nodes__output-labels-desc"
input_file = "experiments/reasoning/qwq-32b/make_hierarchy/qwq-32b-rollouts-labeled.json"

python experiments/reasoning/hierarchical_reasoning/hierarchical_reasoning.py \
    --n_rows 2 \
    --label_level 4 \
    --tree_model_path $tree_path \
    --input_file $input_file \
    --model_name google/gemma-2-27b-it \
    --max_reasoning_steps 15 \
    --llm_framework together \
    --category chess_puzzle \
    --ground_truth_format chess coordinate notation \
    --prompt_verbose \
    --thinking_verbose
```

Example output:

```
PROMPT: 
+ -------------------------------------------------------------------------
| Here is a problem you've been trying to solve: <problem>. 
|
| How will you approach it? Here are some options:
|
| <options>
| "Pawn Dynamics": Analyzing pawn movements and captures to evaluate positional advantages.
| "Tactical Assessment": Analyzing immediate and long-term move consequences.
| "Square Verification": Ensuring accurate board representation through detailed square listing.
| ...
| </options>
+-------------------------------------------------------------------------

>>> I'll select the next thought type: [Tactical Evaluation]

NEXT PROMPT: 
+ -------------------------------------------------------------------------
| Ok, now think about what to do next. Try to follow this approach:
| 
| <approach>
| "Tactical Evaluation": Analyzing different tactical possibilities to refine and select optimal moves.
|</approach>
|
| Here are some examples of how to follow this approach:
|
| <approach_examples>
| Alternatively, perhaps the black rook on e8....
| </approach_examples>
|
| ....
| 
+-------------------------------------------------------------------------

>>> Let's see if the black queen on b8 is attacking any other white pieces.

NEXT PROMPT: 
+ -------------------------------------------------------------------------
| Ok, now what is the next approach you want to take?
|
| <approach>
| "Tactical Evaluation": Analyzing different tactical possibilities to refine and select optimal moves.
| "Square Verification": Ensuring accurate board representation through detailed square listing.
| ...
|</approach>
| 
| Here's what you've thought so far:
|
| <thoughts>
| Tactical Evaluation: Let's see if the black queen on b8 is attacking any other white pieces.
| </thoughts>
| ...
+-------------------------------------------------------------------------
```

--------------------------------

## Experimental Variations

1. **Vanilla Baseline**: Test without hierarchical labeling (just next thoughts)
2. **Level Selection**: Choose the hierarchical level (0-6) for thinking steps
3. **Prompt-tweaking**: Choose different number of examples, iterate on the prompt.
4. **GRPO Training**: Train a model to make hierarchical decisions.

Test each of these with different models, including OpenAI, open-source models, maybe Anthropic.


## Setup

Make sure you have the required dependencies:

```bash
pip install -r requirements.txt
```

(To run OpenAI variations, you'll also need access to OpenAI API: set your `OPENAI_API_KEY` environment variable).

## Usage

### Core Components

The script relies on two main inputs:

1.  **Hierarchical Model (`--tree_model_path`)**: This is the path to the directory containing the pre-computed hierarchical structure (`labeled_hierarchical_tree.gml`), label descriptions, and examples.
2.  **Input Data (`--input_file`)**: This is a JSON file containing the problems to solve (e.g., chess puzzles). Each entry should have a `problem` field and optionally a `ground_truth` answer and `category`.

### Basic Usage

Here's a typical command to run hierarchical reasoning on a few examples using the "together" LLM framework:

```bash
tree_path="experiments/reasoning/qwq-32b/models/chess_agglomerative_clustering_outputs__discretized__labels_descriptions__child-nodes__output-labels-desc"
input_file="experiments/reasoning/qwq-32b/make_hierarchy/qwq-32b-rollouts-labeled.json"

python experiments/reasoning/hierarchical_reasoning/hierarchical_reasoning.py \
    --n_rows 5 \
    --label_level 4 \
    --tree_model_path "$tree_path" \
    --input_file "$input_file" \
    --model_name "google/gemma-2-27b-it" \
    --llm_framework "together" \
    --category "chess_puzzle"
```

### Extended Usage Examples

#### Running with OpenAI

To use an OpenAI model like `gpt-4o-mini`, set your `OPENAI_API_KEY` and run:

```bash
python experiments/reasoning/hierarchical_reasoning/hierarchical_reasoning.py \
    --n_rows 10 \
    --label_level 3 \
    --tree_model_path "$tree_path" \
    --input_file "$input_file" \
    --model_name "gpt-4o-mini" \
    --llm_framework "openai"
```

#### Running with a Local VLLM Server

First, start your VLLM server (see the VLLM setup guide below). Then, run the script pointing to your local server:

```bash
python experiments/reasoning/hierarchical_reasoning/hierarchical_reasoning.py \
    --n_rows 10 \
    --label_level 3 \
    --tree_model_path "$tree_path" \
    --input_file "$input_file" \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --llm_framework "vllm" \
    --vllm_base_url "http://localhost:8000/v1"
```

#### Running the Vanilla Baseline

To test performance without the hierarchical reasoning structure:

```bash
python experiments/reasoning/hierarchical_reasoning/hierarchical_reasoning.py \
    --vanilla_baseline \
    --n_rows 20 \
    --input_file "$input_file" \
    --model_name "gpt-4o-mini" \
    --llm_framework "openai"
```

## Command Line Arguments

### Data and Model Paths
- `--input_file`: Path to the input JSON file with problems.
- `--tree_model_path`: Path to the directory containing the hierarchical model data.

### Reasoning Configuration
- `--vanilla_baseline`: If set, runs the vanilla baseline without hierarchical labeling.
- `--label_level`: The level of the hierarchy to use for thinking steps (0-6, default: 3).
- `--max_reasoning_steps`: Maximum number of reasoning steps per problem (default: 10).
- `--category`: Filter problems by a specific category from the input file.

### LLM Configuration
- `--llm_framework`: The LLM backend to use. Options: `openai`, `vllm`, `together` (default: `openai`).
- `--model_name`: The specific model to use for reasoning (e.g., `gpt-4o-mini`).
- `--temperature`: Temperature for model sampling (default: 0.1).
- `--vllm_base_url`: The base URL for a running VLLM server (used with `--llm_framework vllm`).

### Processing and Caching
- `--n_rows`: Number of examples to process from the input file (default: 10).
- `--cache_dir`: Directory for caching results and examples (default: `experiments/reasoning/hierarchical_reasoning/cache`).

### Verbosity
- `--prompt_verbose`: If set, prints the full prompts sent to the LLM.
- `--thinking_verbose`: If set, prints the intermediate thoughts of the model.

## Output

The script generates several types of output:

1. **Console Output**: Progress updates and final accuracy summary
2. **Examples Cache**: Cached best examples for each thinking pattern (`examples_cache_level_X.pkl`)
3. **Results File**: Detailed results in JSON format (`results_hierarchical_levelX_TIMESTAMP.json` or `results_vanilla_levelX_TIMESTAMP.json`)

### Results Format

Each result entry contains:
```json
{
    "problem_id": "unique_id",
    "problem": "chess problem text",
    "use_hierarchical": true/false,
    "label_level": 3,
    "thought_types": ["Move Analysis", "Tactical Evaluation", "Finish thinking"],
    "thoughts": ["list of generated thoughts"],
    "final_response": "final model response",
    "extracted_answer": "Qg5",
    "is_correct": true/false,
    "ground_truth": "correct answer"
}
```

## Comparison Study Example

To run a comprehensive comparison study:

```bash
# Run vanilla baseline
python experiments/reasoning/hierarchical_reasoning/hierarchical_reasoning.py \
    --vanilla_baseline \
    --n_rows 50 \
    --input_file "$input_file" \
    --cache_dir results/vanilla

# Run hierarchical reasoning at different levels
for level in 0 1 2 3 4 5 6; do
    python experiments/reasoning/hierarchical_reasoning/hierarchical_reasoning.py \
        --label_level $level \
        --n_rows 50 \
        --input_file "$input_file" \
        --cache_dir results/hierarchical
done
```

## Caching Behavior

- **Examples Cache**: Best examples for each thinking pattern are cached per level. This saves time and API calls when running multiple experiments with the same level.
- **Results Cache**: All results are automatically saved with timestamps to prevent overwriting.
- **Cache Directory**: All cache files are stored in the specified cache directory for easy organization.

## Understanding Hierarchical Levels

The hierarchical levels represent different granularities of thinking patterns:

- **Level 6**: Most specific, fine-grained thinking patterns
- **Level 3**: Balanced specificity (default)
- **Level 0**: Most general, high-level thinking patterns

Higher levels provide more specific guidance, while lower levels offer more general directions.

--------------------------------

# VLLM Server Setup Guide in this repo

This guide will help you set up and run your hierarchical reasoning experiments using a local VLLM server instead of OpenAI, allowing you to use 2 GPUs out of 4 on your server.

## Prerequisites

Make sure you have the required packages installed:

```bash
pip install vllm requests torch transformers
```

## Setup

If you prefer manual control:

#### 1. Start VLLM Server
Start the server in the background so you can continue to use your terminal.

```bash
# Start server on GPUs 0,1 with Llama model
python start_vllm_server.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --gpus 0,1 &

# Or start with QwQ model on different port
python start_vllm_server.py --model Qwen/QwQ-32B-Preview --port 8001 --gpus 0,1 &
```

#### 2. Check Server Status
```bash
python check_vllm_server.py --port 8000
```

## Running Your Experiments

### Hierarchical Reasoning Test

The following command is an example of how to run the hierarchical reasoning script with a VLLM-served model. **Remember to define `$tree_path` and `$input_file` as shown in the examples above.**

```bash
python experiments/reasoning/hierarchical_reasoning/hierarchical_reasoning.py \
    --n_rows 10 \
    --label_level 3 \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --temperature 0.1 \
    --llm_framework vllm \
    --vllm_base_url http://localhost:8000/v1 \
    --tree_model_path "$tree_path" \
    --input_file "$input_file" \
    --category chess_puzzle
```

### GRPO Training

The following command is an example of how to run the GRPO training script. **Remember to define `$tree_path` and `$data_file` accordingly.**

```bash
python experiments/reasoning/hierarchical_reasoning/train_model_grpo.py \
    --data_file experiments/reasoning/qwq-32b/make_hierarchy/qwq-32b-rollouts-labeled.json \
    --planner_model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --reasoning_model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --llm_framework vllm \
    --vllm_base_url http://localhost:8000/v1 \
    --output_dir experiments/reasoning/hierarchical_reasoning/checkpoints/planner_grpo \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --tree_model_path "$tree_path" \
    --epochs 3 \
    --max_steps_per_episode 20 \
    --kl_coeff 0.1 \
    --advantage_type binary \
    --num_rollouts_per_prompt 16 \
    --gradient_accumulation_steps 4 \
    --sample_action
```

## Server Management

### Check Server Status
```bash
python check_vllm_server.py --port 8000
```

### Stop Server
You can stop a specific server by port, or stop all running VLLM servers.

```bash
# Stop a specific server
python cleanup_vllm_servers.py --port 8000

# Stop all VLLM servers
python cleanup_vllm_servers.py --kill-all
```

### View Server Logs
Logs are stored in the `logs/` directory with a timestamp.

```bash
# View the latest log file
tail -f logs/vllm_server_*.log
```

## GPU Configuration

The setup uses 2 GPUs by default (GPUs 0 and 1). You can customize this:

```bash
# Use different GPUs
python start_vllm_server.py --gpus 2,3

# Use single GPU
python start_vllm_server.py --gpus 0
```

## Troubleshooting

### Server Won't Start
1. Check if GPUs are available: `nvidia-smi`
2. Check if ports are free: `lsof -i :8000`
3. Check server logs: `cat vllm_server.log`

### Memory Issues
- For large models like QwQ-32B, you may need to reduce `max_model_len`:
```bash
python start_vllm_server.py --model Qwen/QwQ-32B-Preview --max-model-len 8192
```

### Connection Issues
- Make sure the server is running: `python check_vllm_server.py`
- Check if firewall is blocking the port
- Verify the correct base URL in your experiment commands

### Performance Optimization
- Use `--disable-log-requests` to reduce logging overhead (already included in `start_vllm_server.py`)
- Adjust batch size in your experiments based on available memory
- Consider using smaller models for faster iteration during development

## Model Recommendations

### For Development/Testing
- `meta-llama/Meta-Llama-3.1-8B-Instruct` (fits on 2 GPUs easily)
- `microsoft/DialoGPT-medium` (smaller, faster for testing)

### For Production
- `Qwen/QwQ-32B-Preview` (if you have enough GPU memory)
- `meta-llama/Meta-Llama-3.1-70B-Instruct` (requires significant memory)

## Environment Variables

You can set these environment variables for convenience:

```bash
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
export CUDA_VISIBLE_DEVICES="0,1"
```

Then use them in your scripts:
```bash
python experiments/reasoning/hierarchical_reasoning/hierarchical_reasoning.py \
    --llm_framework vllm \
    --model_name $VLLM_MODEL \
    --vllm_base_url $VLLM_BASE_URL \
    # ... other arguments
```

## Files Created

The setup creates these files:
- `logs/vllm_server_...log` - Server logs, created in the `logs` directory with timestamps.

## Next Steps

1. Start the server: `python start_vllm_server.py &`
2. Test with a small number of examples first
3. Monitor GPU usage: `watch -n 1 nvidia-smi`
4. Scale up your experiments once everything is working

## Support

If you encounter issues:
1. Check the server logs: `tail -f logs/vllm_server_*.log`
2. Verify server status: `python check_vllm_server.py`
3. Check GPU usage: `nvidia-smi`
4. Ensure all dependencies are installed: `