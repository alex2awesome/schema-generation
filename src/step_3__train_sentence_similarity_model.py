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

# Set tokenizers parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
transformers_logging.set_verbosity_info()  # Set transformers logging level to INFO

def main(args):
    logging.info("Starting the training process.")

    # 1. Load a model to finetune with
    logging.info(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

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
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        run_name=args.run_name,
        logging_dir='./logs',  # Directory for storing logs
        report_to='all',  # Report to all available integrations (e.g., TensorBoard)
    )

    # 6. Create an evaluator & evaluate the base model
    logging.info("Creating evaluator for initial evaluation.")
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["anchor"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="all-nli-dev",
    )
    if args.do_initial_evaluation:
        logging.info("Performing initial evaluation of the base model.")
        dev_evaluator(model)

    # 7. Create a trainer & train
    logging.info("Initializing the trainer and starting training.")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()
    logging.info("Training completed.")

    # 8. Save the trained model
    logging.info(f"Saving the trained model to {args.output_dir}/trained-model")
    model.save_pretrained(f"{args.output_dir}/trained-model")
    logging.info("Model saved successfully.")


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

    args = parser.parse_args()
    logging.info("Parsed command line arguments.")
    main(args)


    """
    Example of how to run this script locally:

    python src/train_sentence_similarity_model.py \
        --model_name "sentence-transformers/all-MiniLM-L6-v2" \
        --data_file "experiments/editorial/triplets_gpt-4o-mini-similarity-data.jsonl" \
        --output_dir "experiments/editorial/models/editorial-sentence-similarity-model" \
        --num_train_epochs 3 \
        --train_batch_size 8 \
        --eval_batch_size 8 \
        --eval_strategy "epoch" \
        --save_strategy "epoch" \
        --run_name "editorial-sentence-similarity-model" \
        --do_initial_evaluation 

    Make sure to replace "path/to/your/dataset.json" with the actual path to your dataset file.
    """

