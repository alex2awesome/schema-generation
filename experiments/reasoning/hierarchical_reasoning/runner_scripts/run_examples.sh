#!/bin/bash

# Hierarchical Reasoning Example Runner
# This script demonstrates how to run various configurations

echo "=== Hierarchical Reasoning Experiments ==="
echo "This script will run several experiments to compare different configurations."
echo ""

# Create results directory
mkdir -p experiments/reasoning/hierarchical_reasoning/results

# Set common parameters
N_ROWS=20
MODEL_NAME="gpt-4o-mini"
TEMPERATURE=0.1

echo "Running experiments with $N_ROWS examples using $MODEL_NAME..."
echo ""

# 1. Vanilla Baseline
echo "1/8 Running vanilla baseline..."
python experiments/reasoning/hierarchical_reasoning/hierarchical_reasoning.py \
    --vanilla_baseline \
    --n_rows $N_ROWS \
    --model_name $MODEL_NAME \
    --temperature $TEMPERATURE \
    --cache_dir experiments/reasoning/hierarchical_reasoning/results/vanilla

echo "✓ Vanilla baseline completed"
echo ""

# 2. Hierarchical reasoning at different levels
for level in 0 1 2 3 4 5 6; do
    echo "$((level + 2))/8 Running hierarchical reasoning at level $level..."
    python experiments/reasoning/hierarchical_reasoning/hierarchical_reasoning.py \
        --label_level $level \
        --n_rows $N_ROWS \
        --model_name $MODEL_NAME \
        --temperature $TEMPERATURE \
        --cache_dir experiments/reasoning/hierarchical_reasoning/results/level_$level
    
    echo "✓ Level $level completed"
    echo ""
done

echo "=== All experiments completed! ==="
echo ""
echo "Results are saved in: experiments/reasoning/hierarchical_reasoning/results/"
echo ""
echo "To analyze results, you can use:"
echo "  ls experiments/reasoning/hierarchical_reasoning/results/*/results_*.json"
echo ""

# Optional: Run a quick analysis
echo "Quick accuracy summary:"
echo "======================"

for dir in experiments/reasoning/hierarchical_reasoning/results/*/; do
    if [ -d "$dir" ]; then
        config_name=$(basename "$dir")
        latest_result=$(ls -t "$dir"results_*.json 2>/dev/null | head -1)
        if [ -n "$latest_result" ]; then
            # Extract accuracy from the results file
            accuracy=$(python -c "
import json
try:
    with open('$latest_result', 'r') as f:
        results = json.load(f)
    correct = sum(1 for r in results if r.get('is_correct', False))
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    print(f'{accuracy:.1%}')
except:
    print('N/A')
            " 2>/dev/null)
            echo "$config_name: $accuracy"
        fi
    fi
done

echo ""
echo "For detailed analysis, examine the individual JSON result files." 