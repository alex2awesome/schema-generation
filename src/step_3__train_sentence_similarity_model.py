"""
This script trains a sentence similarity model using the SentenceTransformers library.
It loads a pre-trained model, fine-tunes it on a provided dataset, and evaluates its performance.
The dataset can be specified as a single file or a pattern of files. The script also supports
a debug mode to reduce the size of the training dataset for faster iteration during development.

Command line arguments:
- model_name (str): The name of the pre-trained model to load.
- data_file (str, optional): Path to a single dataset file.
- data_file_pattern (str, optional): Pattern to match multiple dataset files.
- test_size (float): Proportion of the dataset to include in the test split.
- debug (bool): Flag to indicate whether to run in debug mode.

Outputs:
- Trained sentence similarity model saved to the specified output directory.
- Evaluation results printed to the console.
"""

import argparse
import logging
import os
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from transformers import logging as transformers_logging
import torch.distributed as dist
import torch.multiprocessing as mp
import torch

# Set tokenizers parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
transformers_logging.set_verbosity_info()  # Set transformers logging level to INFO

def main(args):
    logging.info("Starting the training process.")

    is_distributed = args.local_rank != -1

    if is_distributed:
        # Distributed mode (launched with torchrun or similar)
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
        if not dist.is_initialized():
            # Ensure MASTER_ADDR and MASTER_PORT are set if 'env://', torchrun handles this.
            dist.init_process_group(backend=args.ddp_backend, init_method='env://')
        logging.info(f"Distributed mode: Initialized process group rank {args.local_rank} on {device} with {args.ddp_backend} backend.")
    else:
        # Single-device mode (plain `python` execution)
        logging.info("Running in single-device mode.")
        os.environ["ACCELERATE_DISTRIBUTED_TYPE"] = "NO"  # Crucial for Hugging Face Trainer
        if torch.cuda.is_available():
            # CUDA_VISIBLE_DEVICES is respected by PyTorch.
            # 'cuda' will map to the first available and visible GPU.
            device = torch.device("cuda")
            num_gpus_available = torch.cuda.device_count()
            logging.info(f"Single-device mode: {num_gpus_available} GPU(s) available to PyTorch.")
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                logging.info(f"CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES')}' is set.")
            elif num_gpus_available > 1:
                 logging.warning("Multiple GPUs are visible to PyTorch, but CUDA_VISIBLE_DEVICES is not set. Defaulting to the first visible GPU (typically cuda:0).")
            logging.info(f"Using device: {device}")
        else:
            device = torch.device("cpu")
            logging.info(f"Single-device mode: No CUDA GPUs available. Using device: {device}")
        # args.local_rank is already -1, which is correct for non-distributed Trainer

    # 1. Load a model to finetune with
    logging.info(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    
    # Attempt to enable gradient checkpointing
    try:
        if hasattr(model, '0') and hasattr(model[0], 'auto_model') and hasattr(model[0].auto_model, 'gradient_checkpointing_enable'):
            # logging.info("Attempting to enable gradient checkpointing on the underlying transformer model.")
            # model[0].auto_model.gradient_checkpointing_enable()
            # logging.info("Gradient checkpointing enabled.")
            pass  # Keep the if/else structure, but do nothing here
        elif hasattr(model, '_first_module') and hasattr(model._first_module(), 'auto_model') and hasattr(model._first_module().auto_model, 'gradient_checkpointing_enable'):
            # Another common pattern for accessing the transformer in SBERT
            # logging.info("Attempting to enable gradient checkpointing via _first_module().")
            # model._first_module().auto_model.gradient_checkpointing_enable()
            # logging.info("Gradient checkpointing enabled via _first_module().")
            pass  # Keep the if/else structure, but do nothing here
        else:
            logging.warning("Could not enable gradient checkpointing automatically: Model structure not as expected or method not found.")
    except Exception as e:
        logging.warning(f"Exception while trying to enable gradient checkpointing: {e}")

    # Move model to GPU if available
    model = model.to(device)
    if device.type == 'cuda':  # Clear cache if using any CUDA device
        torch.cuda.empty_cache()  # Clear any existing cache

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Ensuring output directory exists: {args.output_dir}")

    # 3. Load a dataset to finetune on
    if args.data_file:
        logging.info(f"Loading dataset from: {args.data_file}")
        dataset = load_dataset("json", data_files=args.data_file)
    elif args.data_file_pattern:
        logging.info(f"Loading dataset from pattern: {args.data_file_pattern}")
        dataset = load_dataset("json", data_files=args.data_file_pattern)
    else:
        raise ValueError("Either --data_file or --data_file_pattern must be provided.")
    dataset = dataset['train'].train_test_split(test_size=args.test_size)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    logging.info("Dataset loaded and split into train and test sets.")

    # Limit the training dataset to the specified size if needed
    if len(train_dataset) > args.train_subset_size:
        logging.info(f"Limiting training dataset to {args.train_subset_size} samples (from {len(train_dataset)}).")
        train_dataset = train_dataset.select(range(args.train_subset_size))

    # Debug mode: reduce the size of the training dataset
    if args.debug:
        logging.info("Debug mode is ON. Reducing the size of the training dataset.")
        train_dataset = train_dataset.select(range(min(100, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(10, len(eval_dataset))))

    # 4. Define a loss function
    logging.info("Defining the loss function.")
    loss = MultipleNegativesRankingLoss(model)

    # 5. Specify training arguments
    logging.info("Setting up training arguments.")
    training_args_dict = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "batch_sampler": BatchSamplers.NO_DUPLICATES,
        "eval_strategy": args.eval_strategy,
        "eval_steps": args.eval_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "logging_steps": args.logging_steps,
        "run_name": args.run_name,
        "logging_dir": './logs',  # Directory for storing logs
        "report_to": 'all',  # Report to all available integrations (e.g., TensorBoard)
        "local_rank": args.local_rank, # Will be -1 for single device
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    
    if is_distributed:
        training_args_dict["ddp_backend"] = args.ddp_backend
        training_args_dict["ddp_find_unused_parameters"] = True # Keep True for DDP as it was needed
    else:
        # Ensure these are not passed or are None for single-device mode
        training_args_dict["ddp_backend"] = None
        training_args_dict["ddp_find_unused_parameters"] = None

    training_args = SentenceTransformerTrainingArguments(**training_args_dict)

    # 6. Create an evaluator & evaluate the base model
    if args.do_initial_evaluation and (args.local_rank == 0 or args.local_rank == -1):
        logging.info("Creating evaluator for initial evaluation.")
        dev_evaluator = TripletEvaluator(
            anchors=eval_dataset["anchor"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            name="all-nli-dev",
            batch_size=args.eval_batch_size,  # Use smaller batch size for evaluation
            show_progress_bar=(args.local_rank <= 0), # Only show progress bar on main process
        )
        logging.info("Performing initial evaluation of the base model.")
        try:
            with torch.cuda.amp.autocast(enabled=args.fp16):
                dev_evaluator(model)
        except Exception as e:
            logging.error(f"Error during initial evaluation: {e}")
            if args.local_rank != -1:
                dist.destroy_process_group()
            raise e

    # 7. Create a trainer & train
    logging.info("Initializing the trainer and starting training.")
    # Create an evaluator for the trainer to use during training
    dev_evaluator_for_trainer = None
    if args.local_rank <= 0: # Only create evaluator on main process or non-distributed
        dev_evaluator_for_trainer = TripletEvaluator(
            anchors=eval_dataset["anchor"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            name="all-nli-dev-trainer",
            batch_size=args.eval_batch_size,
            show_progress_bar=(args.local_rank <= 0),
        )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator_for_trainer, # Pass evaluator here
    )
    
    try:
        trainer.train()
        logging.info("Training completed.")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        if args.local_rank != -1:
            dist.destroy_process_group()
        raise e

    # 8. Save the trained model (only on main process)
    if args.local_rank in [-1, 0]:
        logging.info(f"Saving the trained model to {args.output_dir}/trained-model")
        model.save_pretrained(f"{args.output_dir}/trained-model")
        logging.info("Model saved successfully.")

    # Clean up distributed training
    if is_distributed and dist.is_initialized(): # Check before destroying
        dist.destroy_process_group()
        logging.info("Destroyed process group.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune a SentenceTransformer model.')
    # Model and dataset parameters
    parser.add_argument(
        '--model_name', type=str, default='microsoft/mpnet-base',
        help='Name of the pre-trained model to fine-tune.'
    )
    parser.add_argument(
        '--data_file', type=str, default=None,
        help='Path to the dataset file.'
    )
    parser.add_argument(
        '--data_file_pattern', type=str, default=None,
        help='Pattern to glob the dataset file with.'
    )
    parser.add_argument(
        '--output_dir', type=str, default='models/mpnet-base-all-nli-triplet',
        help='Output directory for the trained model.'
    )
    # Training parameters
    parser.add_argument('--num_train_epochs', type=int, default=1,
                        help='Number of training epochs.')
    parser.add_argument('--train_batch_size', type=int, default=16,
                        help='Training batch size per device.')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help='Evaluation batch size per device.')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio.')
    parser.add_argument('--fp16', action='store_true',
                        help='Use fp16 training.')
    parser.add_argument('--bf16', action='store_true',
                        help='Use bf16 training.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of updates steps to accumulate before performing a backward pass.')

    # Dataset parameters
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--train_subset_size', type=int, default=100_000,
                        help='Number of training samples to select.')

    # Evaluation and logging parameters
    parser.add_argument('--eval_strategy', type=str, default='steps',
                        help='Evaluation strategy.')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='Evaluation steps.')
    parser.add_argument('--save_strategy', type=str, default='steps',
                        help='Model save strategy.')
    parser.add_argument('--save_steps', type=int, default=100,
                        help='Save steps.')
    parser.add_argument('--save_total_limit', type=int, default=2,
                        help='Total number of model checkpoints to keep.')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Logging steps.')
    parser.add_argument('--run_name', type=str, default='mpnet-base-all-nli-triplet',
                        help='Run name for logging.')

    # Other parameters
    parser.add_argument('--do_initial_evaluation', action='store_true',
                        help='Evaluate the base model before training.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode to reduce dataset size for faster debugging.')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training. Automatically set by torchrun.')
    parser.add_argument('--ddp_backend', type=str, default='nccl', 
                        help='DDP backend to use (e.g., nccl, gloo) if running distributed.')

    args = parser.parse_args()
    main(args)


    """
    Example of how to run this script locally:

CUDA_VISIBLE_DEVICES=1 python src/step_3__train_sentence_similarity_model.py \
    --model_name "sentence-transformers/all-MiniLM-L6-v2" \
    --data_file "experiments/hate-speech/triplets_vllm-similarity-data.jsonl" \
    --output_dir "experiments/hate-speech/models/hate-speech-sentence-similarity-model" \
    --num_train_epochs 3 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --run_name "hate-speech-sentence-similarity-model" \
    --do_initial_evaluation 

------------------------------------------------------------------------------------------------
To run distributed training, use the following command:
------------------------------------------------------------------------------------------------

accelerate launch --num_processes 4 --deepspeed_config_file ds_config.json src/step_3__train_sentence_similarity_model.py \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --data_file experiments/hate-speech/triplets_vllm-similarity-data.jsonl \
    --output_dir experiments/hate-speech/models/hate-speech-sentence-similarity-model \
    --train_batch_size 32 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --eval_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --fp16 \
    --do_initial_evaluation


accelerate launch --num_processes 4 --deepspeed_config_file ds_config.json src/step_3__train_sentence_similarity_model.py \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --data_file experiments/reasoning/qwq-32b/triplets_self-similarity-data-full.jsonl \
    --output_dir experiments/reasoning/qwq-32b/models/sentence-similarity-model \
    --train_batch_size 32 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --eval_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --fp16 \
    --do_initial_evaluation    

------------------------------------------------------------------------------------------------
To run single-process training with torchrun, use the following command:
------------------------------------------------------------------------------------------------
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=1 --master_port=29501 src/step_3__train_sentence_similarity_model.py \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --data_file experiments/news-discourse/triplets_vllm-similarity-data.jsonl \
    --output_dir experiments/news-discourse/models/news-discourse-sentence-similarity-model \
    --train_batch_size 32 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --eval_batch_size 32 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 2 \
    --fp16 \
    --do_initial_evaluation    

"""

