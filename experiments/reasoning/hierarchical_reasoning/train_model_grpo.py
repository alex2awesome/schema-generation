#!/usr/bin/env python3
"""
train_model_grpo.py

Train a hierarchical planner LM using GRPO to select high‐level reasoning steps (h_n).
Supports both "mean‐normalized" and "binary" reward‐advantage objectives.

In more detail:
- `collect_episodes(...)` runs "episodes" in which:
    - A frozen hierarchical reasoner (fixed LLM) takes a problem prompt and, step by step:
        - Uses the planner model to pick a high-level reasoning step label (`h_n`) by sampling or greedy argmax over a small label vocabulary
        - Feeds that label into the fixed reasoner LM to generate low-level reasoning text
        - Repeats until "Finish thinking" or a max number of steps
    - Each high-level step's prompt and log-prob are cached as a `StepRecord`
    - At episode end, the reasoner's final answer is extracted and compared to ground truth to yield a binary reward
- `compute_advantages(...)` calculates GRPO-style advantages:
    - "binary": advantage = (reward - batch_mean_reward)
    - "mean":    advantage = (reward - batch_mean) / batch_std
- `grpo_update(...)` implements the GRPO policy update for the planner:
    - For each recorded step:
        - Re-evaluate the planner on the cached prompt to get `log_new`
        - Compute a policy-gradient term: `-advantage[episode_idx] * log_new`
        - Compute a KL term: `old_log_prob - log_new`
    - Averages over all steps to form:
        - `policy_loss = mean(-adv * log_new)`
        - `kl_loss = mean(old_log_prob - log_new)`
        - `total_loss = policy_loss + kl_coeff * kl_loss`
    - Backpropagates only through the planner model; the reasoning LM remains frozen
- The `main()` training loop:
    - Iterates over epochs, drawing batches of problems
    - Calls `collect_episodes()` to gather step records and rewards
    - Computes advantages with `compute_advantages()`
    - Updates planner weights via `grpo_update()`
    - Logs average reward and periodically saves checkpoints

Theory behind the code:
- We treat high-level reasoning steps ("choose next kind of thought") as a discrete policy πθ(h_n | state).
- GRPO (Guided Reward Policy Optimization) uses batch-wise normalization of binary rewards (Bernoulli outcomes) to compute advantages, avoiding a learned value baseline.
- Each planner step is stored to (re)evaluate π_new(h_n | s_n) at update time, enabling an explicit KL penalty between the old and new policy.
- By freezing the large reasoning LLM and only training the smaller planner, we leverage a strong, pretrained reasoner while learning a lightweight, interpretable high-level decision policy under reinforcement feedback.

Usage example (on server):
  python experiments/reasoning/hierarchical_reasoning/train_model_grpo.py \
    --data_file experiments/reasoning/qwq-32b/make_hierarchy/qwq-32b-rollouts-labeled.json \
    --planner_model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --reasoning_model_name gpt-4o-mini \
    --output_dir experiments/reasoning/hierarchical_reasoning/checkpoints/planner_grpo \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --tree_model_path experiments/reasoning/qwq-32b/models/chess_agglomerative_clustering_outputs__discretized__labels_descriptions__child-nodes__output-labels-desc \
    --epochs 3 \
    --max_steps_per_episode 20 \
    --kl_coeff 0.1 \
    --advantage_type binary \
    --num_rollouts_per_prompt 16 \
    --gradient_accumulation_steps 4 \
    --sample_action
"""

import argparse
import json
import os
import random
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('train_model_grpo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the existing HierarchicalReasoner class and utilities
# (Assumes that your project's PYTHONPATH includes the parent directory of "experiments")
from hierarchical_reasoning import HierarchicalReasoner
from reasoning_prompts import (
    CHOOSE_FIRST_NODE,
    CHOOSE_NEXT_NODE,
    FOLLOW_HIGH_LEVEL_START,
    FOLLOW_HIGH_LEVEL_CONTINUATION,
    FOLLOW_HIGH_LEVEL_FINAL,
)


class ChessReasoningDataset(Dataset):
    """
    A simple Dataset wrapping a list of JSON objects, each containing at least:
      - "prompt": the problem statement
      - ground truth answer fields as in HierarchicalReasoner
    """
    def __init__(self, data_df):
        self.examples = data_df[["prompt", "ground_truth_json"]].assign(
            answer=lambda df: df["ground_truth_json"].str.get("move_list")
        ).to_dict(orient="records")
        logger.info(f"Loaded {len(self.examples)} examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class StepRecord:
    """
    One planner action, used later for GRPO update.
    Added fields:
      - example_idx: which prompt in the current batch
      - action_index: index within candidate_ids (so we can locate it in the label‐distribution)
      - old_candidate_log_probs: full log‐probs over the label set under π_old
      - advantage: the scalar advantage assigned to this entire episode
    """
    prompt_text: str                    # The text prompt used for this step
    action_token_id: int                # the single token ID chosen
    action_index: int                   # index within the candidate_ids array
    old_log_prob: torch.Tensor          # scalar tensor = log π_old(a|s)
    old_candidate_log_probs: torch.Tensor  # shape: (num_labels,) log π_old(v|s) for each label v
    episode_idx: int                    # rollout index (0..R−1) within this prompt
    example_idx: int                    # prompt index (0..batch_size−1) within this collect_episodes call
    advantage: torch.Tensor = None      # to be filled after grouping; scalar tensor

    def to(self, device: torch.device) -> 'StepRecord':
        """Move all tensors to specified device."""
        return StepRecord(
            prompt_text=self.prompt_text,
            action_token_id=self.action_token_id,
            action_index=self.action_index,
            old_log_prob=self.old_log_prob.to(device),
            old_candidate_log_probs=self.old_candidate_log_probs.to(device),
            episode_idx=self.episode_idx,
            example_idx=self.example_idx,
            advantage=self.advantage.to(device) if self.advantage is not None else None,
        )


def _nan_check(logits: torch.Tensor, input_ids: torch.Tensor, first_prompt: str = None):
    if torch.isnan(logits).any():
        logger.warning(f"NaN values in logits during collection!")
        logger.warning(f"Input shape: {input_ids.shape}")
        logger.warning(f"Logits shape: {logits.shape}")
        logger.warning(f"Number of NaN values: {torch.isnan(logits).sum().item()}")
        if first_prompt:
            logger.warning(f"Input text: {first_prompt[:200]}...")  # Print first 200 chars
        logger.warning(f"Input token IDs: {input_ids[0][:10]}...")  # Print first 10 tokens


def _choose_next_action_with_planner(
    planner_model: AutoModelForCausalLM,
    planner_tokenizer: AutoTokenizer,
    reasoner: HierarchicalReasoner,
    base_options_df: pd.DataFrame,
    label_token_ids: List[tuple],
    candidate_ids: torch.Tensor,
    temperature: float,
    sample_action: bool,
    device: torch.device,
    problem_text: str,
    step_idx: int,
    example_idx: int,
    episode_idx: int,
    thought_history: List[str],
    low_level_text_history: List[str],
) -> (str, StepRecord):
    """
    Choose next action with planner model.
    """
    thinking_history = "\n".join(
        f"Thought type {tt}. \t Thought: {txt}"
        for tt, txt in zip(thought_history, low_level_text_history)
    )
    reasoning_options_df = base_options_df.sample(frac=1).reset_index(drop=True)
    options_str = "\n".join(reasoning_options_df["formatted_description"].tolist())
    if step_idx == 0:
        prompt = CHOOSE_FIRST_NODE.format(problem=problem_text, options=options_str)
    else:
        prompt = CHOOSE_NEXT_NODE.format(
            problem=problem_text,
            thinking=thinking_history,
            options=options_str,
        )

    # get planner model inputs
    inputs = planner_tokenizer(prompt, return_tensors="pt").to(device)

    # get logits for candidate labels
    with torch.no_grad():
        out = planner_model(**inputs)                     # (1, seq_len, vocab_size)
        logits = out.logits[:, -1, :]                     # (1, vocab_size)
        _nan_check(logits, inputs.input_ids, prompt)
        # Restrict to our label‐tokens:
        candidate_logits = logits[0, candidate_ids]       # (num_labels,)

        # Compute full log‐prob distribution over the labels under π_old:
        candidate_log_probs = torch.log_softmax(candidate_logits, dim=-1)  # (num_labels,)

        # Sample or Argmax:
        if sample_action:
            probs = torch.softmax(candidate_logits / temperature, dim=-1)
            dist = torch.distributions.Categorical(probs)
            chosen_idx = dist.sample().item()
            log_old = dist.log_prob(torch.tensor(chosen_idx, device=device))
        else:
            chosen_idx = torch.argmax(candidate_logits).item()
            log_old = candidate_log_probs[chosen_idx]

    # Identify the chosen label string and its token_id:
    chosen_label, chosen_token_id = label_token_ids[chosen_idx]

    # Build StepRecord, storing FULL old‐log‐prob vector + scalar old_log of the chosen action:
    step_record = StepRecord(
        prompt_text=prompt,
        action_token_id=chosen_token_id,
        action_index=chosen_idx,
        old_log_prob=log_old.detach().cpu(),
        old_candidate_log_probs=candidate_log_probs.detach().cpu(),
        episode_idx=episode_idx,
        example_idx=example_idx,
        advantage=None,
    )

    return chosen_label, step_record


def _generate_next_thought_text(
    problem_text: str,
    options_df: pd.DataFrame,
    chosen_label: str,
    step_idx: int,
    thought_history: List[str],
    low_level_text_history: List[str],
    reasoner: HierarchicalReasoner,
) -> str:
    """
    Generate next thought text.
    """
    formatted_thinking_history = "\n".join(
        f"Thought type {tt}. \t Thought: {txt}"
        for tt, txt in zip(thought_history, low_level_text_history)
    )

    options_df = options_df.sample(frac=1).reset_index(drop=True)
    row_opt = options_df.loc[lambda df: df["label"] == chosen_label].iloc[0]
    examples_for_label = row_opt["examples"]
    formatted_desc = row_opt["formatted_description"]

    if step_idx == 0:
        low_prompt = FOLLOW_HIGH_LEVEL_START.format(
            problem=problem_text,
            approach=formatted_desc,
            approach_examples="\n\n".join(examples_for_label) if examples_for_label else "",
        )
    elif chosen_label == "Finish thinking":
        low_prompt = FOLLOW_HIGH_LEVEL_FINAL.format(
            problem=problem_text,
            thinking=formatted_thinking_history,
        )
    else:
        low_prompt = FOLLOW_HIGH_LEVEL_CONTINUATION.format(
            problem=problem_text,
            thinking=formatted_thinking_history,
            approach=formatted_desc,
            approach_examples="\n\n".join(examples_for_label) if examples_for_label else "",
        )

    low_output = reasoner.llm(prompt=low_prompt)
    return low_output


def collect_episodes(
    planner_model: AutoModelForCausalLM,
    planner_tokenizer: AutoTokenizer,
    reasoner: HierarchicalReasoner,
    base_options_df: pd.DataFrame,
    label_token_ids: List[tuple],
    args: argparse.Namespace,
    device: torch.device,
    dataset_batch: List[Dict[str, Any]],
) -> (List[StepRecord], List[float]):
    """
    For each problem in dataset_batch, run multiple episodes (rollouts):
      - Use the planner_model to sample a sequence of high‐level steps (h_n),
      - For each h_n, call the fixed reasoning LM inside HierarchicalReasoner to generate low‐level tokens,
      - Terminate when "Finish thinking" or max_steps_per_episode,
      - Compute a multi‐component reward (correctness + length penalty).
    Returns:
      - step_records: List[StepRecord], one per high‐level step (advantage field is filled)
      - episode_rewards: List[float], one per episode
    """
    planner_model.eval()
    all_step_records: List[StepRecord] = []
    episode_rewards: List[float] = []

    # Pre‐extract candidate_ids tensor for faster indexing
    candidate_ids = torch.tensor(
        [tok_id for (_, tok_id) in label_token_ids],
        device=device
    )

    # To pass unchanged into the helper:
    base_kwargs = dict(
        planner_model=planner_model,
        planner_tokenizer=planner_tokenizer,
        reasoner=reasoner,
        base_options_df=base_options_df,
        label_token_ids=label_token_ids,
        candidate_ids=candidate_ids,
        device=device,
        temperature=args.temperature,
        sample_action=args.sample_action,
    )

    # Run multiple rollouts per prompt
    R = args.num_rollouts_per_prompt
    batch_size = len(dataset_batch)

    for example_idx, example in enumerate(dataset_batch):
        problem_text = example["prompt"]

        for episode_idx in range(R):
            step_idx = 0
            thought_history: List[str] = []
            low_level_text_history: List[str] = []

            # --- STEP 0: Choose first node ---
            chosen_label, step_record = _choose_next_action_with_planner(
                **base_kwargs,
                problem_text=problem_text,
                step_idx=step_idx,
                example_idx=example_idx,
                episode_idx=episode_idx,
                thought_history=thought_history,
                low_level_text_history=low_level_text_history,
            )
            thought_history.append(chosen_label)
            all_step_records.append(step_record)

            # Low‐level reasoning for step 0
            low_output = _generate_next_thought_text(
                problem_text=problem_text,
                options_df=base_options_df,
                chosen_label=chosen_label,
                step_idx=step_idx,
                thought_history=thought_history,
                low_level_text_history=low_level_text_history,
                reasoner=reasoner,
            )
            low_level_text_history.append(low_output)

            # --- STEPS 1 ... max_steps_per_episode - 1 ---
            for step_idx in range(1, args.max_steps_per_episode):
                chosen_label, step_record = _choose_next_action_with_planner(
                    **base_kwargs,
                    problem_text=problem_text,
                    step_idx=step_idx,
                    example_idx=example_idx,
                    episode_idx=episode_idx,
                    thought_history=thought_history,
                    low_level_text_history=low_level_text_history,
                )
                all_step_records.append(step_record)

                # Low‐level reasoning for this step
                low_output = _generate_next_thought_text(
                    problem_text=problem_text,
                    options_df=base_options_df,
                    chosen_label=chosen_label,
                    step_idx=step_idx,
                    thought_history=thought_history,
                    low_level_text_history=low_level_text_history,
                    reasoner=reasoner,
                )
                low_level_text_history.append(low_output)

                if chosen_label == "Finish thinking":
                    break

            # Episode ends: compute multi‐component reward
            final_text = low_level_text_history[-1]
            pred_answer = reasoner._extract_answer(final_text)
            is_correct, ground_truth = reasoner._check_answer_correctness(pred_answer, example)
            r_correct = 1.0 if is_correct else 0.0

            # Length reward: shorter low‐level text gets higher reward
            total_chars = sum(len(txt) for txt in low_level_text_history)
            r_length = 1.0 / (1.0 + total_chars)

            reward = r_correct + args.length_reward_weight * r_length
            episode_rewards.append(reward)

    # ===== GROUP‐WISE ADVANTAGE NORMALIZATION =====
    # We know: episode_rewards is length = batch_size * R, ordered by (example_idx, episode_idx)
    rewards_tensor = torch.tensor(episode_rewards, dtype=torch.float32)
    rewards_tensor = rewards_tensor.view(batch_size, R)  # shape: (batch_size, R)

    mu = rewards_tensor.mean(dim=1, keepdim=True)  # (batch_size, 1)
    if args.advantage_type == "mean":
        sigma = rewards_tensor.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        adv_tensor = (rewards_tensor - mu) / sigma  # shape: (batch_size, R)
    else:  # "binary"
        adv_tensor = rewards_tensor - mu            # shape: (batch_size, R)

    adv_flat = adv_tensor.view(-1)  # shape: (batch_size * R,)

    # Attach advantage to each StepRecord (loop over all_step_records):
    # Each step_record has fields example_idx (0..batch_size−1) and episode_idx (0..R−1).
    for rec in all_step_records:
        global_idx = rec.example_idx * R + rec.episode_idx
        rec.advantage = adv_flat[global_idx].detach().cpu()

    return all_step_records, episode_rewards


def grpo_update(
    planner_model: AutoModelForCausalLM,
    planner_tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step_records: List[StepRecord],
    accumulated_rewards: List[float],
    kl_coeff: float,
    epsilon: float,
    device: torch.device,
    gradient_accumulation_steps: int,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Update the planner_model parameters using GRPO with:
      - A PPO‐style clipped surrogate, using rec.advantage
      - A full KL over the label distribution per step
    """

    planner_model.train()
    pg_terms = []
    kl_terms = []

    for rec in step_records:
        # Move tensors to GPU for this step
        rec = rec.to(device)

        # Retokenize the prompt text
        inputs = planner_tokenizer(rec.prompt_text, return_tensors="pt").to(device)
        out = planner_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        logits = out.logits[:, -1, :]                                 # shape (1, vocab_size)
        _nan_check(logits, inputs.input_ids)

        # Reconstruct candidate_ids
        candidate_ids = torch.tensor(
            [tok_id for (_, tok_id) in label_token_ids],
            device=device
        )

        candidate_logits_new = logits[0, candidate_ids]               # shape: (num_labels,)
        new_candidate_log_probs = torch.log_softmax(candidate_logits_new, dim=-1)  # shape: (num_labels,)

        # 1) PPO‐style SURROGATE for the chosen action:
        old_lp = rec.old_candidate_log_probs.to(device)               # (num_labels,)
        action_idx = rec.action_index
        log_new = new_candidate_log_probs[action_idx]                 # scalar
        log_old = old_lp[action_idx]                                  # scalar

        ratio_act = torch.exp(log_new - log_old)                       # π_new(a) / π_old(a)
        clipped_ratio_act = torch.clamp(ratio_act, 1.0 - epsilon, 1.0 + epsilon)

        adv = rec.advantage.to(device)  # scalar advantage

        pg_term = -torch.min(ratio_act * adv, clipped_ratio_act * adv)
        pg_terms.append(pg_term)

        # 2) FULL KL over all labels v:
        old_probs = torch.exp(old_lp)                                  # shape: (num_labels,)
        kl_term = (old_probs * (old_lp.to(device) - new_candidate_log_probs)).sum()
        kl_terms.append(kl_term)

        # Clean up
        del out, logits, candidate_logits_new, new_candidate_log_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pg_loss = torch.stack(pg_terms).mean()
    kl_loss = torch.stack(kl_terms).mean()
    total_loss = pg_loss + kl_coeff * kl_loss

    total_loss = total_loss / gradient_accumulation_steps
    total_loss.backward()

    # Check for NaN gradients before clipping
    has_nan_grad = False
    nan_params_before_clipping, nan_params_before_optimizer = [], []
    for name, param in planner_model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            has_nan_grad = True
            nan_params_before_clipping.append(name)

    # Clip gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(planner_model.parameters(), max_norm=max_grad_norm)

    # Check for NaN parameters before optimizer step
    has_nan_param = False
    for name, param in planner_model.named_parameters():
        if torch.isnan(param).any():
            has_nan_param = True
            nan_params_before_optimizer.append(name)

    if has_nan_grad or has_nan_param:
        logger.warning(f"NaN gradients in {nan_params_before_clipping}")
        logger.warning(f"NaN params in {nan_params_before_optimizer}")
        logger.warning(f"Skipping optimizer step. Grad norm: {grad_norm}")
        optimizer.zero_grad()
        return float('nan')

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return total_loss.item()


def main():
    parser = argparse.ArgumentParser(description="Train hierarchical planner with GRPO")
    parser.add_argument(
        "--data_file", type=str, required=True,
        help="JSON file with list of reasoning problems"
    )
    parser.add_argument(
        "--planner_model_name", type=str, required=True,
        help="Huggingface model name or path for planner (e.g., gpt2-medium)"
    )
    parser.add_argument(
        "--reasoning_model_name", type=str, required=True,
        help="Model name for the fixed reasoning LM (e.g., gpt-4o-mini)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save planner checkpoints"
    )
    parser.add_argument(
        "--tree_model_path", type=str, required=True,
        help="Path to hierarchical model data (labels, tree, etc.)"
    )
    parser.add_argument(
        "--cache_dir", type=str, default="cache/",
        help="Cache directory for HierarchicalReasoner examples"
    )
    parser.add_argument(
        "--label_level", type=int, default=3,
        help="Hierarchy level for thought labels"
    )
    parser.add_argument(
        "--llm_framework", choices=["openai", "vllm"], default="openai",
        help="Framework for the fixed reasoning LLM"
    )
    parser.add_argument(
        "--ground_truth_format", type=str, default="chess coordinate notation",
        help="Format string for correctness check prompt"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Number of prompts per batch (total episodes = batch_size * num_rollouts_per_prompt)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5,
        help="LR for planner policy optimizer"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of passes over the dataset"
    )
    parser.add_argument(
        "--max_steps_per_episode", type=int, default=10,
        help="Max hierarchical steps in one episode"
    )
    parser.add_argument(
        "--kl_coeff", type=float, default=0.1,
        help="Coefficient for KL regularization term"
    )
    parser.add_argument(
        "--advantage_type", choices=["binary", "mean"], default="binary",
        help="How to compute advantages from binary rewards"
    )
    parser.add_argument(
        "--sample_action", action="store_true",
        help="If set, sample actions from planner; else use argmax."
    )
    parser.add_argument(
        "--vllm_base_url", type=str, default="http://localhost:8000/v1",
        help="Base URL for VLLM server (only used when llm_framework is 'vllm')"
    )
    parser.add_argument(
        "--vllm_gpus", type=str, default="0,1",
        help="GPUs to use for VLLM server (comma-separated, only used when llm_framework is 'vllm')"
    )
    parser.add_argument(
        "--vllm_auto_start", action="store_true", default=True,
        help="Automatically start VLLM server if not running (only used when llm_framework is 'vllm')"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature for planner LM"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--category", type=str, default="chess_puzzle",
        help="Category of problems to use"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
        help="Number of batches to accumulate gradients over before updating"
    )
    parser.add_argument(
        "--num_rollouts_per_prompt", type=int, default=4,
        help="Number of rollouts per prompt for better gradient estimates"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0,
        help="Maximum gradient norm for clipping"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.2,
        help="Clipping epsilon (e.g. 0.2) for PPO‐style surrogate"
    )
    parser.add_argument(
        "--length_reward_weight", type=float, default=0.0,
        help="Weight for the length‐based reward (shorter is better)"
    )
    args = parser.parse_args()
    set_seed(args.seed)

    # Warn if using multiple rollouts with deterministic action selection
    if args.num_rollouts_per_prompt > 1 and not args.sample_action:
        logger.warning("Using multiple rollouts per prompt with deterministic action selection.")
        logger.warning("Consider setting --sample_action for diversity in rollouts.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load planner tokenizer & model
    planner_tokenizer = AutoTokenizer.from_pretrained(args.planner_model_name)
    planner_model = AutoModelForCausalLM.from_pretrained(args.planner_model_name).to(device)

    # If tokenizer has no pad token, set it to eos_token
    if planner_tokenizer.pad_token_id is None:
        planner_tokenizer.pad_token_id = planner_tokenizer.eos_token_id

    # Instantiate a single HierarchicalReasoner and build base reasoning options once
    reasoner_args = argparse.Namespace(
        model_path=args.tree_model_path,
        label_level=args.label_level,
        vanilla_baseline=False,
        llm_framework=args.llm_framework,
        category=args.category,
        model_name=args.reasoning_model_name,
        temperature=args.temperature,
        cache_dir=args.cache_dir,
        input_file=args.data_file,
        ground_truth_format=args.ground_truth_format,
        vllm_base_url=args.vllm_base_url,
        vllm_gpus=args.vllm_gpus,
        vllm_auto_start=args.vllm_auto_start,
    )
    reasoner = HierarchicalReasoner(reasoner_args)
    base_options_df = reasoner._setup_reasoning_options()

    # Prepare dataset & dataloader
    input_data = pd.read_json(args.data_file, lines=False)
    dataset = ChessReasoningDataset(input_data)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: batch  # returns list of examples
    )

    # Extract label list and convert each to a single token for the planner
    labels = base_options_df["label"].tolist()
    new_label_tokens = [f" {lbl}" for lbl in labels]
    num_added = planner_tokenizer.add_tokens(new_label_tokens)
    logger.info(f"Added {num_added} new tokens for labels.")
    planner_model.resize_token_embeddings(len(planner_tokenizer))

    # Build label_token_ids for quick reference
    label_token_ids = []
    for lbl in labels:
        token_ids = planner_tokenizer.encode(f" {lbl}", add_special_tokens=False)
        if len(token_ids) == 1:
            label_token_ids.append((lbl, token_ids[0]))
    if not label_token_ids:
        raise RuntimeError("Failed to add label tokens to tokenizer.")

    # Optimizer & scheduler for planner
    optimizer = torch.optim.AdamW(planner_model.parameters(), lr=args.learning_rate)

    # Fix scheduler calculation to account for gradient accumulation
    batches_per_epoch = len(dataset) // args.batch_size
    updates_per_epoch = batches_per_epoch // args.gradient_accumulation_steps
    total_updates = updates_per_epoch * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_updates // 10),
        num_training_steps=total_updates,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0

    for epoch in range(args.epochs):
        logger.info(f"\n===== Epoch {epoch+1}/{args.epochs} =====")

        # Initialize accumulation variables
        accumulated_step_records: List[StepRecord] = []
        accumulated_rewards: List[float] = []

        for batch_idx, batch_examples in enumerate(dataloader):
            # Collect episodes for this batch (multiple rollouts per prompt):
            step_records, episode_rewards = collect_episodes(
                planner_model=planner_model,
                planner_tokenizer=planner_tokenizer,
                reasoner=reasoner,
                base_options_df=base_options_df,
                label_token_ids=label_token_ids,
                args=args,
                device=device,
                dataset_batch=batch_examples,
            )

            # Accumulate across gradient_accumulation_steps
            accumulated_step_records.extend(step_records)
            accumulated_rewards.extend(episode_rewards)

            # When it's time to update:
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                loss = grpo_update(
                    planner_model=planner_model,
                    planner_tokenizer=planner_tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step_records=accumulated_step_records,
                    accumulated_rewards=accumulated_rewards,
                    kl_coeff=args.kl_coeff,
                    epsilon=args.epsilon,
                    device=device,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    max_grad_norm=args.max_grad_norm,
                )

                # Logging
                avg_reward = sum(accumulated_rewards) / len(accumulated_rewards)
                num_episodes = len(accumulated_rewards)
                num_prompts = num_episodes // args.num_rollouts_per_prompt
                logger.info(
                    f"Step {global_step:4d} | Loss {loss:.4f} | "
                    f"AvgReward {avg_reward:.2f} | "
                    f"Episodes: {num_episodes} (from {num_prompts} prompts)"
                )
                global_step += 1

                # Reset accumulation
                accumulated_step_records = []
                accumulated_rewards = []

        # Handle any remaining accumulated data at end of epoch
        if accumulated_step_records:
            loss = grpo_update(
                planner_model=planner_model,
                planner_tokenizer=planner_tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                step_records=accumulated_step_records,
                accumulated_rewards=accumulated_rewards,
                kl_coeff=args.kl_coeff,
                epsilon=args.epsilon,
                device=device,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
            )

            avg_reward = sum(accumulated_rewards) / len(accumulated_rewards)
            num_episodes = len(accumulated_rewards)
            num_prompts = num_episodes // args.num_rollouts_per_prompt
            logger.info(
                f"Final step {global_step:4d} | Loss {loss:.4f} | "
                f"AvgReward {avg_reward:.2f} | "
                f"Episodes: {num_episodes} (from {num_prompts} prompts)"
            )
            global_step += 1

        # Save checkpoint at end of epoch
        ckpt_path = Path(args.output_dir) / f"planner_epoch_{epoch+1}.pt"
        torch.save(planner_model.state_dict(), ckpt_path)
        logger.info(f"Saved planner checkpoint to {ckpt_path}")

    # Final save
    final_path = Path(args.output_dir) / "planner_final.pt"
    torch.save(planner_model.state_dict(), final_path)
    logger.info(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()