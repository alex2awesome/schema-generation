#!/usr/bin/env python3
"""
Hierarchical Reasoning Script

This script implements and tests a hierarchical reasoning process on a given set of problems,
such as chess puzzles. It compares a structured, multi-step reasoning approach against a
standard "vanilla" baseline.

Core Functionality:
-------------------
The script operates in one of two modes:

1.  **Hierarchical Reasoning Mode (Default)**:
    - For each problem, the script engages in a multi-step reasoning process.
    - In each step, it first uses an LLM to select a high-level "thought pattern" from a
      pre-defined hierarchy (e.g., "Analyze threats," "Evaluate pawn structure").
    - It then prompts the LLM again to generate a detailed thought that follows the chosen pattern,
      often providing few-shot examples of that pattern.
    - This loop continues for a configurable number of steps or until the model decides to
      "Finish thinking" and provide a final answer.

2.  **Vanilla Baseline Mode (`--vanilla_baseline`)**:
    - For each problem, the script sends a single, direct prompt to the LLM to solve the
      problem without any hierarchical guidance.

The script is designed to be flexible, supporting various LLM backends like OpenAI,
TogetherAI, and a local VLLM server.

Outputs:
--------
All outputs are saved to the directory specified by `--cache_dir` (default: 'cache/').

- **JSON Results**: A JSON file is created for each run, containing the problem,
  the full sequence of thoughts, the final answer, and correctness information.
  Filename format: `results_{mode}_level{level}_{model}_{timestamp}.json`

- **Examples Cache**: Few-shot examples are selected via an LLM for crafting the lower-level prompts
  and getting the LLM to reason via a certain approach. To speed up subsequent runs, the examples 
  are cached in a `.pkl` file.
  Filename format: `examples_cache_level_{level}_n_{num_icl_examples}.pkl`

- **Log File**: A log file named `hierarchical_reasoning.log` is created in the script's
  directory, capturing the execution flow and any warnings or errors.

How to Run:
-----------
Before running, ensure all dependencies from `requirements.txt` are installed.

Example 1: Running hierarchical reasoning with a local VLLM model.
```bash
# Define paths to your data
tree_path="experiments/reasoning/qwq-32b/models/chess_agglomerative_clustering_outputs__discretized__labels_descriptions__child-nodes__output-labels-desc"
input_file="experiments/reasoning/qwq-32b/make_hierarchy/qwq-32b-rollouts-labeled.json"

python experiments/reasoning/hierarchical_reasoning/hierarchical_reasoning.py \
    --n_rows 10 \
    --label_level 3 \
    --tree_model_path "$tree_path" \
    --input_file "$input_file" \
    --category "chess_puzzle" \
    --llm_framework vllm \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --vllm_base_url "http://localhost:8000/v1"
```

Example 2: Running the vanilla baseline with OpenAI.
```bash
# Define paths to your data
input_file="experiments/reasoning/qwq-32b/make_hierarchy/qwq-32b-rollouts-labeled.json"

python experiments/reasoning/hierarchical_reasoning/hierarchical_reasoning.py \
    --n_rows 20 \
    --vanilla_baseline \
    --input_file "$input_file" \
    --category "chess_puzzle" \
    --llm_framework openai \
    --model_name "gpt-4o-mini"
```
"""

import argparse
import pandas as pd
import json
import os
import sys
import logging
from pathlib import Path
import random

# Add project root to path to allow absolute imports
here = Path(__file__).parent.resolve()
project_root = here.parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import Dict, List, Any, Optional
import pickle
from tqdm.auto import tqdm
import re
import time
from llm_interface import get_llm_interface, VLLMInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('hierarchical_reasoning.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(project_root / 'src'))
import utils_trees


# Import prompts and LLM interface
from reasoning_prompts import (
    CHOOSE_EXAMPLES_PROMPT,
    LOW_LEVEL_THINKING_START,
    LOW_LEVEL_THINKING_CONTINUATION,
    LOW_LEVEL_THINKING_FINAL,
    CHOOSE_FIRST_THOUGHT_TYPE,
    CHOOSE_NEXT_THOUGHT_TYPE,
    VANILLA_BASELINE,
    make_labeling_structure,
    make_best_examples_structure
)


class HierarchicalReasoner:
    """Main class for hierarchical reasoning on chess problems"""
    
    def __init__(self, args):
        self.args = args
        self.tree_model_path = args.tree_model_path
        self.label_level = args.label_level
        self.use_hierarchical = not args.vanilla_baseline
        self.cache_dir = Path(args.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.num_icl_examples = args.num_icl_examples
        
        # Initialize LLM interface
        self.llm_framework = args.llm_framework
        self.model_name = args.model_name # main LLM
        self.high_level_llm_framework = args.high_level_llm_framework
        self.high_level_model_name = args.high_level_model_name # high-level LLM
        self.temperature = args.temperature
        self.vllm_base_url = args.vllm_base_url
        self.vllm_gpus = args.vllm_gpus
        self.vllm_auto_start = args.vllm_auto_start
        self.vllm_verbose = args.start_vllm_verbose
        self.thinking_verbose = args.thinking_verbose
        self.prompt_verbose = args.prompt_verbose
        self.is_first_problem = True  # Add state for first problem
        self.llm = get_llm_interface(
            llm_framework=self.llm_framework,
            model_name=self.model_name,
            temperature=self.temperature,
            vllm_base_url=self.vllm_base_url, # only needed for vllm
            vllm_gpus=self.vllm_gpus, # only needed for vllm
            vllm_auto_start=self.vllm_auto_start, # only needed for vllm
            verbose=self.vllm_verbose
        )
        if self.model_name == self.high_level_model_name: # if the model is the same for high and low level, use the same LLM
            self.high_level_llm = self.llm
        else:
            self.high_level_llm = get_llm_interface(
                llm_framework=self.high_level_llm_framework,
                model_name=self.high_level_model_name,
                temperature=self.temperature,
                vllm_base_url=self.vllm_base_url, # only needed for vllm
                vllm_gpus=self.vllm_gpus, # only needed for vllm,
                vllm_auto_start=self.vllm_auto_start, # only needed for vllm
                verbose=self.vllm_verbose
            )
        
        # Load data
        self._load_data()
        
        # Setup examples cache
        self.examples_cache_file = self.cache_dir / f"examples_cache_level_{self.label_level}_n_{self.num_icl_examples}.pkl"
        self.examples_cache = self._load_examples_cache()
        
        # Results storage
        self.results = []
        
    def _load_data(self):
        """Load hierarchical tree data and input examples"""
        logger.info("Loading hierarchical data...")
        
        # Load tree and labels
        tree_file = f'{self.tree_model_path}/labeled_hierarchical_tree.gml'
        labels_file = f'{self.tree_model_path}/labels_and_descriptions.csv'
        
        self.labeled_tree = utils_trees.load_hierarchical_tree(tree_file)
        self.labels_and_descriptions = pd.read_csv(labels_file)
        
        # Load examples for the specified level
        examples_file = f"{self.tree_model_path}/labeled_chunks__level-{self.label_level}.csv"
        self.examples_df = pd.read_csv(examples_file, index_col=0)
        
        # Load input chess problems
        self.input_data = pd.read_json(self.args.input_file)
        
        # Filter by category only if specified
        if self.args.category and 'category' in self.input_data.columns:
            logger.info(f"Filtering for category: {self.args.category}")
            self.input_data = self.input_data.loc[lambda df: df['category'] == self.args.category]
        elif 'category' in self.input_data.columns:
            logger.info(f"Available categories: {self.input_data['category'].unique().tolist()}")
            
        if 'ground_truth_json' in self.input_data.columns:
            self.input_data['ground_truth'] = self.input_data['ground_truth_json'].apply(json.loads)
        
        logger.info(f"Loaded {len(self.input_data)} chess problems")
        
    def _load_examples_cache(self) -> Dict[str, List[str]]:
        """Load cached examples or return empty dict"""
        if self.examples_cache_file.exists():
            logger.info(f"Loading examples cache from {self.examples_cache_file}")
            with open(self.examples_cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            logger.info(f"No examples cache found at {self.examples_cache_file}, creating new one...")
        return {}
    
    def _save_examples_cache(self):
        """Save examples cache to disk"""
        if self.examples_cache and len(self.examples_cache) > 0 and (self.examples_cache_file is not None):
            logger.info(f"Saving examples cache to {self.examples_cache_file}")
            with open(self.examples_cache_file, 'wb') as f:
                pickle.dump(self.examples_cache, f)
 
    def _choose_best_examples(self, thought_pattern: str) -> List[str]:
        """Choose best examples for a thought pattern, using cache if available"""
        if thought_pattern in self.examples_cache:
            return self.examples_cache[thought_pattern]
        
        logger.info(f"Choosing best examples for: {thought_pattern}")
        
        # Get examples for this thought pattern
        examples = (
            self.examples_df
            .loc[lambda df: df['chunks'].str.split().str.len() < 400]
            .loc[lambda df: df['chunks'].str.split().str.len() > 50] 
            .loc[lambda df: df['labels'] == thought_pattern, 'chunks']
        )
        
        if len(examples) == 0:
            logger.warning(f"No examples found for {thought_pattern}")
            return []
        
        # Sample examples for choosing best ones
        sampled_examples = examples.sample(min(30, len(examples))).tolist()
        examples_text = '\n\n'.join(sampled_examples)

        num_to_select = self.num_icl_examples * 3

        # Get label description
        label_description = (
            self.labels_and_descriptions
            .loc[lambda df: df['level'] == self.label_level]
            .loc[lambda df: df['label'] == thought_pattern]
            .apply(lambda x: f'"{x["label"]}": {x["description"]}', axis=1)
        )
        
        if len(label_description) == 0:
            logger.warning(f"No description found for {thought_pattern}")
            return sampled_examples[:num_to_select]
        
        label_description = label_description.iloc[0]
        
        # Choose best examples using LLM
        prompt = CHOOSE_EXAMPLES_PROMPT.format(
            thought_pattern=label_description,
            examples=examples_text,
            num_examples=num_to_select
        )
        try:
            response = self.high_level_llm(
                model_name=self.high_level_model_name,
                prompt=prompt,
                response_format=make_best_examples_structure(num_examples=num_to_select)
            )
            
            # Handle both structured and unstructured responses
            if hasattr(response, 'best_examples') or ('best_examples' in response):
                best_examples = response.best_examples
            else:
                logger.warning("Structured output failed for BestExamples, using string response fallback")
                # Try to parse the string response manually
                best_examples = sampled_examples[:num_to_select]  # Fallback to first 5
                
        except Exception as e:
            logger.error(f"Error choosing examples for {thought_pattern}: {e}")
            best_examples = sampled_examples[:num_to_select]
        
        # Cache the result
        self.examples_cache[thought_pattern] = best_examples
        return best_examples
    
    def _setup_reasoning_options(self):
        """Setup the reasoning options for the current level.
        
        todo: we can actually save more examples than we need in the examples cache, and then cheaply call this 
        function to get a downsampling of the examples for each label. Come back to this if you see problems in training/inference.
        """
        # Add "Finish thinking" option
        finished_row = pd.Series({
            'label': 'Finish thinking', 
            'description': 'Finish thinking about the problem and answer.'
        }).to_frame().T
        
        # Get formatted options
        formatted_options = (
            self.labels_and_descriptions
            .loc[lambda df: df['level'] == self.label_level]
            .pipe(lambda df: pd.concat([df, finished_row]))
            .assign(formatted_description=lambda df: df.apply(
                lambda x: f'"{x["label"]}": {x["description"]}', axis=1
            ))
            .sample(frac=1)  # Shuffle
        )
        
        # Get best examples for each option
        examples_map = {}
        for _, row in formatted_options.iterrows():
            if row['label'] != 'Finish thinking':
                # this step caches the examples for the label in self.examples_cache
                examples_map[row['label']] = self._choose_best_examples(row['label'])
            else:
                examples_map[row['label']] = []
        
        # Add examples to dataframe
        formatted_options['examples'] = formatted_options['label'].map(examples_map)
        self._save_examples_cache()
        return formatted_options
    
    def _get_prompts(self):
        """Get all the prompt templates"""
        return {
            'low_level_thinking_start': LOW_LEVEL_THINKING_START,
            'low_level_thinking_continuation': LOW_LEVEL_THINKING_CONTINUATION,
            'low_level_thinking_final': LOW_LEVEL_THINKING_FINAL,
            'choose_first_thought_type': CHOOSE_FIRST_THOUGHT_TYPE,
            'choose_next_thought_type': CHOOSE_NEXT_THOUGHT_TYPE,
            'vanilla_baseline': VANILLA_BASELINE
        }
    
    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract chess move answer from text"""
        # Look for boxed answer first
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        # Look for answer tags
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if answer_match:
            # Look for boxed inside answer tags
            answer_content = answer_match.group(1).strip()
            inner_boxed = re.search(r'\\boxed\{([^}]+)\}', answer_content)
            if inner_boxed:
                return inner_boxed.group(1).strip()
            return answer_content
        
        return None
    
    def _check_answer_correctness(self, predicted_answer, row) -> bool:
        """Check if the predicted answer is correct"""        
        # Get ground truth from the data
        # You may need to adjust this based on your data format
        if 'answer' in row:
            ground_truth = str(row['answer']).strip()
        elif 'correct_answer' in row:
            ground_truth = str(row['correct_answer']).strip()
        elif 'ground_truth' in row:
            fen, ground_truth = row['ground_truth']['fen'], row['ground_truth']['move_list']
        else:
            # If no ground truth available, we can't check
            return False, None
        
        if predicted_answer is None:
            return False, ground_truth

        # Normalize both answers for comparison
        response = self.high_level_llm(
            model_name=self.high_level_model_name,
            prompt=f"""Is the following answer equivalent to the ground truth(s) for {self.args.ground_truth_format}? Answer with "yes" or "no".
            Predicted: {predicted_answer}
            Ground: {ground_truth}
            """,
            temperature=self.temperature
        )
        return response.lower().strip() == "yes", ground_truth
    
    def _run_vanilla_baseline(self, problem: str) -> Dict[str, Any]:
        """Run vanilla baseline without hierarchical labels"""
        prompts = self._get_prompts()
        
        try:
            response = self.llm(
                model_name=self.model_name,
                prompt=prompts['vanilla_baseline'].format(problem=problem),
                temperature=self.temperature
            )
            
            answer = self._extract_answer(response)
            
            return {
                'thought_types': [],
                'thoughts': [response],
                'final_response': response,
                'extracted_answer': answer
            }
        except Exception as e:
            logger.error(f"Error in vanilla baseline: {e}")
            return {
                'thought_types': [],
                'thoughts': [f"Error: {e}"],
                'final_response': f"Error: {e}",
                'extracted_answer': None
            }
    
    def _run_hierarchical_reasoning(self, problem: str, reasoning_options: pd.DataFrame) -> Dict[str, Any]:
        """Run hierarchical reasoning with structured thought selection"""
        prompts = self._get_prompts()
        
        # Create label choices format
        label_choices_format = make_labeling_structure(reasoning_options['label'].tolist())
        
        all_thoughts = []
        thought_types = []
        
        try:
            # Get first thought type
            starting_prompt = prompts['choose_first_thought_type'].format(
                problem=problem, 
                options='\n'.join(reasoning_options['formatted_description'].sample(frac=1))
            )
            
            if self.prompt_verbose and self.is_first_problem:
                logging.info(f"--- PROMPT (Initial high-level thought) ---\n{starting_prompt}")
            
            r = self.llm(
                model_name=self.model_name,
                prompt=starting_prompt,
                response_format=label_choices_format,
                temperature=self.temperature
            )
            
            # Handle both structured and unstructured responses
            if hasattr(r, 'next_thought_type') or ('next_thought_type' in r):
                next_thought_type = r.next_thought_type
            else:
                # Fallback: try to parse the string response
                logger.warning("Structured output failed, attempting to parse string response")
                # Use first available option as fallback
                available_options = reasoning_options['label'].tolist()
                next_thought_type = available_options[0] if available_options else "Finish thinking"
                logger.warning(f"Using fallback thought type: {next_thought_type}")
            thought_types.append(next_thought_type)
            
            if self.args.thinking_verbose:
                logging.info(f"I'll select the first thought type: {next_thought_type}\n")

            # Get examples and formatted description for first thought type
            option_row = reasoning_options.loc[lambda df: df['label'] == next_thought_type].iloc[0]
            superset_examples = option_row['examples']
            if superset_examples and len(superset_examples) > self.num_icl_examples:
                examples_to_show = random.sample(superset_examples, self.num_icl_examples)
            else:
                examples_to_show = superset_examples
            formatted_desc = option_row['formatted_description']
            
            starting_low_level_prompt = prompts['low_level_thinking_start'].format(
                problem=problem,
                approach=formatted_desc,
                approach_examples='\n\n'.join(examples_to_show) if examples_to_show else ''
            )
            
            if self.prompt_verbose and self.is_first_problem:
                logging.info(f"--- PROMPT (Initial thought) ---\n{starting_low_level_prompt}")

            thoughts = self.llm(
                model_name=self.model_name,
                prompt=starting_low_level_prompt,
                temperature=self.temperature
            )
            all_thoughts.append(thoughts)
            if self.args.thinking_verbose:
                logging.info(f"Initial thought: {thoughts}")
            
            # Iterate through reasoning steps
            for step_idx in range(self.args.max_reasoning_steps):
                if step_idx  ==  self.args.max_reasoning_steps - 1:
                    next_thought_type = 'Finish thinking'

                else:
                    # Format thinking so far
                    thought_format = [
                        f'{thought_type} thought type: {thought}'
                        for thought_type, thought in zip(thought_types, all_thoughts)
                    ]
                    
                    next_thought_type_prompt = prompts['choose_next_thought_type'].format(
                        problem=problem,
                        thinking='\n'.join(thought_format),
                        options='\n'.join(reasoning_options['formatted_description'].sample(frac=1))
                    )
                    
                    if self.prompt_verbose and self.is_first_problem:
                        logging.info(f"--- PROMPT (Next high-level thought) ---\n{next_thought_type_prompt}")
                    
                    r = self.llm(
                        model_name=self.model_name,
                        prompt=next_thought_type_prompt,
                        response_format=label_choices_format,
                        temperature=self.temperature
                    )
                    
                    # Handle both structured and unstructured responses
                    if (hasattr(r, 'next_thought_type')) or ('next_thought_type' in r):
                        next_thought_type = r.next_thought_type
                    else:
                        # Fallback: try to parse the string response
                        logger.warning("Structured output failed in continuation, attempting to parse string response")
                        # Use first available option as fallback
                        available_options = reasoning_options['label'].tolist()
                        next_thought_type = available_options[0] if available_options else "Finish thinking"
                        logger.warning(f"Using fallback thought type: {next_thought_type}")
                
                    if self.args.thinking_verbose:
                        logging.info(f"I'll select the next thought type: {next_thought_type}\n")

                thought_types.append(next_thought_type)
                
                # Get examples and formatted description for next thought type
                option_row = reasoning_options.loc[lambda df: df['label'] == next_thought_type].iloc[0]
                superset_examples = option_row['examples']
                if superset_examples and len(superset_examples) > self.num_icl_examples:
                    examples_to_show = random.sample(superset_examples, min(self.num_icl_examples, len(superset_examples)))
                else:
                    examples_to_show = superset_examples
                formatted_desc = option_row['formatted_description']
                
                # Choose prompt type based on whether we're finishing
                if next_thought_type == 'Finish thinking':
                    if self.args.thinking_verbose:
                        logging.info(f"Finishing thinking for {problem}...")
                    continuation_prompt = prompts['low_level_thinking_final']
                    next_low_level_prompt = continuation_prompt.format(
                        problem=problem,
                        thinking='\n'.join(thought_format),
                    )
                else:
                    continuation_prompt = prompts['low_level_thinking_continuation']
                    next_low_level_prompt = continuation_prompt.format(
                        problem=problem,
                        thinking='\n'.join(thought_format),
                        approach=formatted_desc,
                        approach_examples='\n\n'.join(examples_to_show) if examples_to_show else ''
                    )
                
                if self.prompt_verbose and self.is_first_problem:
                    logging.info(f"--- PROMPT (Next thought) ---\n{next_low_level_prompt}")

                thoughts = self.llm(
                    model_name=self.model_name,
                    prompt=next_low_level_prompt,
                    temperature=self.temperature
                )
                all_thoughts.append(thoughts)

                if self.args.thinking_verbose:
                    logging.info(f"\nThought: {thoughts}\n")
                
                if next_thought_type == 'Finish thinking':
                    break
            
            # Extract final answer
            final_response = all_thoughts[-1] if all_thoughts else ""
            answer = self._extract_answer(final_response)
            
            return {
                'thought_types': thought_types,
                'thoughts': all_thoughts,
                'final_response': final_response,
                'extracted_answer': answer
            }
            
        except Exception as e:
            logger.error(f"Error in hierarchical reasoning: {e}")
            return {
                'thought_types': [],
                'thoughts': [f"Error: {e}"],
                'final_response': f"Error: {e}",
                 'extracted_answer': None
            }
    
    def run_single_example(self, row: pd.Series, reasoning_options: pd.DataFrame = None) -> Dict[str, Any]:
        """Run reasoning on a single chess problem"""
        problem = row['prompt']
        
        if self.use_hierarchical:
            reasoning_options = self._setup_reasoning_options() if reasoning_options is None else reasoning_options
            result = self._run_hierarchical_reasoning(problem, reasoning_options)
        else:
            result = self._run_vanilla_baseline(problem)
        
        # Check answer correctness
        is_correct, ground_truth = self._check_answer_correctness(result['extracted_answer'], row)
        
        # Compile full result
        full_result = {
            'problem_id': row.name,
            'problem': problem,
            'use_hierarchical': self.use_hierarchical,
            'label_level': self.label_level,
            'thought_types': result['thought_types'],
            'thoughts': result['thoughts'],
            'final_response': result['final_response'],
            'extracted_answer': result['extracted_answer'],
            'is_correct': is_correct,
            'ground_truth': ground_truth
        }
        
        return full_result
    
    def run_all_examples(self):
        """Run reasoning on all specified examples"""
        logger.info(f"Running reasoning on {self.args.n_rows} data points...")
        logger.info(f"Using hierarchical: {self.use_hierarchical}")
        if self.use_hierarchical:
            logger.info(f"Label level: {self.label_level}")
        
        # Setup reasoning options once if using hierarchical
        reasoning_options = None
        if self.use_hierarchical:
            reasoning_options = self._setup_reasoning_options()
            logger.info(f"Setup {len(reasoning_options)} reasoning options")
        
        # Sample the examples to run
        examples_to_run = self.input_data.head(self.args.n_rows)
        
        # Run reasoning on each example
        for idx, (_, row) in enumerate(tqdm(
            examples_to_run.iterrows(), desc="Processing examples", total=len(examples_to_run))
        ):
            # try:
            result = self.run_single_example(row, reasoning_options)
            self.is_first_problem = False
            self.results.append(result)
            
            # Log progress
            if (idx + 1) % 5 == 0:
                correct_count = sum(1 for r in self.results if r['is_correct'])
                accuracy = correct_count / len(self.results) if self.results else 0
                logger.info(f"Progress: {idx + 1}/{len(examples_to_run)}, "
                           f"Accuracy so far: {accuracy:.2%}")
            
            # except Exception as e:
            #     logger.error(f"Error processing example {idx}: {e}")
            #     # Add error result
            #     self.results.append({
            #         'problem_id': row.name,
            #         'problem': row['prompt'],
            #         'use_hierarchical': self.use_hierarchical,
            #         'label_level': self.label_level,
            #         'thought_types': [],
            #         'thoughts': [f"Error: {e}"],
            #         'final_response': f"Error: {e}",
            #         'extracted_answer': None,
            #         'is_correct': False,
            #         'ground_truth': row.get('answer', row.get('correct_answer', None))
            #     })
    
    def save_results(self):
        """Save results to cache directory"""
        # Save examples cache
        self._save_examples_cache()
        
        # Create results filename
        mode = "hierarchical" if self.use_hierarchical else "vanilla"
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.cache_dir / f"results_{mode}_level{self.label_level}_{self.model_name.replace('/', '-')}_{timestamp}.json"
        
        output = {
            'model_name': self.model_name,
            'label_level': self.label_level,
            'use_hierarchical_reasoning': self.use_hierarchical,
            'results': self.results
        }
        # Save results
        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        
        # Log summary
        correct_count = sum(1 for r in self.results if r['is_correct'])
        total_count = len(self.results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        logger.info(f"\nSummary:")
        logger.info(f"Total examples: {total_count}")
        logger.info(f"Correct answers: {correct_count}")
        logger.info(f"Accuracy: {accuracy:.2%}")
        logger.info(f"Mode: {'Hierarchical' if self.use_hierarchical else 'Vanilla baseline'}")
        if self.use_hierarchical:
            logger.info(f"Label level: {self.label_level}")


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Reasoning on Chess Problems")
    
    # Model and data paths
    parser.add_argument(
        "--tree_model_path",
        type=str,
        default=None,
        help="Path to the hierarchical tree model data"
    )
    
    # Reasoning configuration
    parser.add_argument(
        "--vanilla_baseline",
        action="store_true",
        help="Run vanilla baseline without hierarchical labeling"
    )
    
    parser.add_argument(
        "--label_level",
        type=int,
        default=3,
        help="Level to use for picking thinking steps (0-6)"
    )
    
    # Processing configuration
    parser.add_argument(
        "--n_rows",
        type=int,
        default=10,
        help="Number of chess examples to process"
    )
    
    parser.add_argument(
        "--max_reasoning_steps",
        type=int,
        default=10,
        help="Maximum number of reasoning steps"
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Category of reasoning problems we'll be using. If not provided, we'll use all categories."
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input file to use for reasoning"
    )

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for main process."
    )

    parser.add_argument(
        "--high_level_model_name",
        type=str,
        default="gpt-4o",
        help="Model to use for non-central tasks where we need precision (specifically, choosing examples, validating answers)."
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for model sampling"
    )
    
    # LLM framework configuration
    parser.add_argument(
        "--llm_framework",
        type=str,
        default="openai",
        choices=["openai", "vllm", "together"],
        help="LLM framework to use"
    )

    parser.add_argument(
        "--high_level_llm_framework",
        type=str,
        default="openai",
        choices=["openai", "vllm", "together"],
        help="LLM framework to use for high-level reasoning"
    )
    
    parser.add_argument(
        "--ground_truth_format",
        type=str,
        default="chess coordinate notation",
        help="The type of answer we expect for the ground truth"
    )
    
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for VLLM server (only used when llm_framework is 'vllm')"
    )
    
    parser.add_argument(
        "--vllm_gpus",
        type=str,
        default="0,1,2,3",
        help="GPUs to use for the VLLM server, if auto-started."
    )
    
    parser.add_argument(
        "--vllm_auto_start",
        action="store_true",
        default=True,
        help="Automatically start VLLM server if not running (only used when llm_framework is 'vllm')"
    )

    parser.add_argument(
        "--start_vllm_verbose",
        action="store_true",
        help="Show detailed logs from the VLLM server when it is auto-started. Default is quiet."
    )

    parser.add_argument(
        "--thinking_verbose",
        action="store_true",
        help="Log all thoughts generated during the reasoning process."
    )

    parser.add_argument(
        "--prompt_verbose",
        action="store_true",
        help="Log all prompts sent to the LLM for the first problem."
    )

    # Cache configuration
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.join(here, "cache"),
        help="Directory for caching results and examples"
    )
    
    parser.add_argument(
        "--num_icl_examples",
        type=int,
        default=5,
        help="Number of in-context examples to use in prompts. A superset of twice this number will be selected by the LLM."
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if (args.label_level < 0) or (args.label_level > 6):
        raise ValueError("label_level must be between 0 and 6")
    
    # Create and run reasoner
    reasoner = HierarchicalReasoner(args)
    reasoner.run_all_examples()
    reasoner.save_results()


if __name__ == "__main__":
    main() 


"""
python experiments/reasoning/hierarchical_reasoning/hierarchical_reasoning.py \
    --n_rows 10 \
    --label_level 3 \
    --model_name gpt-4o-mini \
    --temperature 0.1 \
    --llm_framework openai \
    --tree_model_path experiments/reasoning/qwq-32b/models/chess_agglomerative_clustering_outputs__discretized__labels_descriptions__child-nodes__output-labels-desc


python experiments/reasoning/hierarchical_reasoning/hierarchical_reasoning.py \
    --input_file experiments/reasoning/qwq-32b/make_hierarchy/qwq-32b-rollouts-labeled.json \
    --n_rows 100 \
    --label_level 3 \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --temperature 0.1 \
    --max_reasoning_steps 15 \
    --category "chess_puzzle" \
    --ground_truth_format "chess coordinate notation" \
    --llm_framework vllm \
    --vllm_base_url http://localhost:8000/v1 \
    --vllm_gpus 2,3 \
    --vllm_auto_start \
    --tree_model_path experiments/reasoning/qwq-32b/models/chess_agglomerative_clustering_outputs__discretized__labels_descriptions__child-nodes__output-labels-desc
"""





## things to analyze (beyond raw performance):
## 1. how similar are the reasoning steps to other reasoning steps in this category? Can we induce different categories of reasoning steps?
## 2. how similar are the reasoning steps to the original question? Do we not deviate from the original question?
