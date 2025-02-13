"""
This script runs generic prompts through a given model and input data.
It uses the VLLM library to generate responses and saves the results to a file.

Command line arguments:
- model: The name of the model to use.
- input_data_file: The path to the input data file.
- start_idx: The starting index of the batch to process.
- end_idx: The ending index of the batch to process.
- id_col: The name of the column containing the IDs.
- prompt_col: The name of the column containing the prompts.
- output_file: The path to the output file.
- prompt: A prompt to prepend to each prompt.

Outputs:
- A file containing the responses to the prompts.
"""

from datasets import load_from_disk
import pandas as pd
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import unicodedata
import os, json
import torch
import logging
import random
from vllm import LLM,  SamplingParams
from transformers import AutoTokenizer
from utils_vllm_client import load_model, write_to_file


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

here = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f'{here}/../../config.json'))
os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

BATCH_SIZE = 500


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # meta-llama/Llama-3.1-8B-Instruct
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument('--input_data_file', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--id_col', type=str, default=None)
    parser.add_argument('--prompt_col', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--prompt', type=str, default=None)

    args = parser.parse_args()
    article_df = pd.read_csv(args.input_data_file, index_col=0)

    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None:
        args.end_idx = len(article_df)
    
    # load the model
    sampling_params = SamplingParams(temperature=0.1, max_tokens=4096)
    tokenizer, model = load_model(args.model)
    num_batches = (args.end_idx - args.start_idx) // BATCH_SIZE
    batch_indices = [(i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, args.end_idx)) for i in range(num_batches)]
    random.shuffle(batch_indices)

    for start_idx, end_idx in tqdm(batch_indices):
        dirname = os.path.dirname(args.output_file)
        if (dirname != '') and not os.path.exists(dirname):
            os.makedirs(dirname)

        out_dirname, out_fname = os.path.split(args.output_file)
        fname, fext = os.path.splitext(out_fname)
        output_fname = f'{out_dirname}/{fname}__{start_idx}_{end_idx}{fext}'
        if not os.path.exists(output_fname):
            logging.info(f"Running prompts for batch {start_idx} to {end_idx}")
            # create an empty file to indicate that this batch is being processed
            with open(output_fname, 'w') as f:
                f.write('')

            df = article_df.iloc[start_idx:end_idx]
            prompts_to_run = df[args.prompt_col].tolist()
            if args.prompt is not None:
                prompts_to_run = list(map(lambda x: f"{args.prompt}\n\n{x}", prompts_to_run))

            outputs = model.generate(prompts_to_run, sampling_params)
            write_to_file(output_fname, df[args.id_col], outputs)
        else:
            logging.info(f"Skipping batch {start_idx} to {end_idx} as it already exists")