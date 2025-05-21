"""
This script runs generic prompts through a given model and input data.
It uses either VLLM or OpenAI to generate responses and saves the results to a file.

Command line arguments:
- model: The name of the model to use.
- input_data_file: The path to the input data file.
- start_idx: The starting index of the batch to process.
- end_idx: The ending index of the batch to process.
- id_col: The name of the column containing the IDs.
- prompt_col: The name of the column containing the prompts.
- output_file: The path to the output file.
- prompt: A prompt to prepend to each prompt.
- use_openai: Flag to use OpenAI instead of VLLM.
- debug_mode: Flag to stop after creating batch files for inspection.
- temp_dir: Directory to store temporary batch files.
- multi_sentence: Flag to use multi-sentence processing.
- num_sents_per_prompt: Number of sentences to process per prompt (default: 8).

Outputs:
- A file containing the responses to the prompts.
"""

from datasets import load_from_disk
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import unicodedata
import os, json
import torch
import logging
import random
import spacy
from prompts import (
    EDITORIAL_INITIAL_LABELING_PROMPT, 
    MULTI_SENTENCE_EDITORIAL_LABELING_PROMPT,
    EditorialLabelingResponse,
    MultiSentenceLabelingResponse
)

# Import VLLM only if needed
def load_vllm_dependencies():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    from transformers import AutoTokenizer
    from utils_vllm_client import load_model as load_vllm_model, write_to_file, robust_parse_outputs, check_output_validity
    return LLM, SamplingParams, AutoTokenizer, load_vllm_model, write_to_file, robust_parse_outputs, check_output_validity

# Import OpenAI only if needed
def load_openai_dependencies():
    from utils_openai_client import load_model as load_openai_model, generate_responses as generate_openai_responses
    return load_openai_model, generate_openai_responses

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

here = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(f'{here}/../config.json'):
    config_data = json.load(open(f'{here}/../config.json'))
    os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

def make_labeling_prompts_editorials(input_df, multi_sentence=False, num_sents_per_prompt=8):
    if not multi_sentence:
        all_responses = []
        g = input_df.groupby('doc_index')
        for doc_index, df in g:
            df = df[['sent_index', 'sentence_text']].sort_values('sent_index')
            doc_text = df['sentence_text'].str.cat(sep=' ')
            for _, (sent_index, sentence_text) in df.iterrows():
                all_responses.append({
                    'index': f'{doc_index.replace("/", "_")}__sent_index-{sent_index}',
                    'doc_text': doc_text,
                    'sent_text': sentence_text,
                })

        prompt_df = pd.DataFrame(all_responses)
        prompt_df['prompt'] = (
            prompt_df.apply(lambda r: EDITORIAL_INITIAL_LABELING_PROMPT.format(article=r['doc_text'], sentence=r['sent_text']), axis=1)
        )
        return prompt_df
    else:
        all_responses = []
        g = input_df.groupby('doc_index')
        for doc_index, df in g:
            df = df[['sent_index', 'sentence_text']].sort_values('sent_index')
            doc_text = df['sentence_text'].str.cat(sep=' ')
            sents = df['sentence_text'].tolist()
            
            for i in range(0, len(sents), num_sents_per_prompt):
                sents_chunk = sents[i: i + num_sents_per_prompt]
                sents_chunk = list(map(lambda x: f"(idx {i + x[0]}) {x[1].replace('\n', ' ').strip()}", enumerate(sents_chunk)))
                sents_chunk_text = '\n'.join(sents_chunk)
                
                all_responses.append({
                    'index': f'{doc_index.replace("/", "_")}__chunk-{i}',
                    'doc_text': doc_text,
                    'sentences': sents,
                    'sents_chunk': sents_chunk_text,
                    'num_sents': len(sents_chunk),
                })

        prompt_df = pd.DataFrame(all_responses)
        prompt_df['prompt'] = (
            prompt_df.apply(lambda r: MULTI_SENTENCE_EDITORIAL_LABELING_PROMPT.format(
                k=r['num_sents'],
                article=r['doc_text'],
                sentences=r['sents_chunk']
            ), axis=1)
        )
        return prompt_df

def process_batch_vllm(model, prompts_to_run, response_format):
    """Process a batch of prompts using VLLM."""
    logging.info(f'sample prompt: {prompts_to_run["prompt"].tolist()[0]}')
    json_schema = response_format.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(response_format=json_schema)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024, guided_decoding=guided_decoding_params)

    outputs = model.generate(prompts_to_run['prompt'].tolist(), sampling_params)
    inputted_prompts = list(map(lambda x: x.prompt, outputs))
    model_outputs = list(map(lambda x: x.outputs[0].text, outputs))
    return inputted_prompts, model_outputs, outputs

def process_all_openai(model, prompts, model_name, batch_size, debug_mode, temp_dir, num_sents_per_prompt, batch_type, response_format):
    """Process all prompts using OpenAI's batch API."""
    model_outputs = generate_openai_responses(
        client=model,
        prompt_ids=prompts['index'].tolist(),
        prompts=prompts['prompt'].tolist(),
        model_name=model_name,
        temperature=0.1,
        max_tokens=1024,
        batch_size=batch_size,
        debug_mode=debug_mode,
        temp_dir=temp_dir,
        batch_type=batch_type,
        response_format=response_format
    )

    output_df = pd.DataFrame(model_outputs)
    output_df = output_df.merge(prompts, left_on='custom_id', right_on='index')
    if num_sents_per_prompt > 1:
        output_df['response'] = output_df['response'].apply(lambda x: sorted(x['sentences'], key=lambda y: y['sentence_idx']))
        output_df['sentences'] = output_df.apply(lambda x: list(map(lambda y: x['sentences'][y['sentence_idx']], x['response'])), axis=1)
        output_df = output_df[['custom_id', 'sentences', 'response']]
        full_exp_df = output_df.explode(['sentences', 'response'])
        output_df = pd.concat([full_exp_df.reset_index(drop=True), pd.DataFrame(full_exp_df['response'].tolist())], axis=1).drop(columns=['response'])
    return output_df

def save_outputs(output_fname, model_outputs):
    """Save the outputs to a file."""
    if model_outputs is None:  # Debug mode stopped execution
        logging.info("Debug mode: Stopped after creating batch files")
        return False

    logging.info(f'output: {model_outputs.sample(min(10, len(model_outputs)))}')
    model_outputs.to_json(output_fname, orient='records', lines=True)
    return True

def check_existing_outputs(output_fname, start_idx, end_idx):
    """Check if we already have results for this range."""
    if not os.path.exists(output_fname):
        return False
    
    try:
        existing_df = pd.read_json(output_fname, orient='records', lines=True)
        if len(existing_df) >= (end_idx - start_idx):
            logging.info(f"Found existing results in {output_fname} with {len(existing_df)} rows")
            return True
    except Exception as e:
        logging.warning(f"Error reading existing file {output_fname}: {e}")
    return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # meta-llama/Meta-Llama-3.1-8B-Instruct
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument('--input_data_file', type=str, default=None)
    parser.add_argument('--experiment', type=str, default='editorials')
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--window_size', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--use_openai', action='store_true', help='Use OpenAI instead of VLLM')
    parser.add_argument('--debug_mode', action='store_true', help='Stop after creating batch files for inspection')
    parser.add_argument('--temp_dir', type=str, default=None, help='Directory to store temporary batch files')
    parser.add_argument('--num_sents_per_prompt', type=int, default=8, help='Number of sentences to process per prompt')

    args = parser.parse_args()
    if '.json' in args.input_data_file:
        input_df = pd.read_json(args.input_data_file, orient='records', lines=True)
    elif '.csv' in args.input_data_file:
        input_df = pd.read_csv(args.input_data_file)
    else:
        raise ValueError(f'Input data file {args.input_data_file} must be a json or csv file')

    # make prompts
    prompt_cache_fname = f'{args.output_file}-prompt-cache.json'
    if os.path.exists(prompt_cache_fname):
        prompts = pd.read_json(prompt_cache_fname, orient='records', lines=True)
    else:
        if args.experiment == 'editorials':
            prompts = make_labeling_prompts_editorials(input_df=input_df, multi_sentence=args.num_sents_per_prompt > 1, num_sents_per_prompt=args.num_sents_per_prompt)
        else:
            raise ValueError(f'Experiment {args.experiment} not supported')
        prompts.to_json(prompt_cache_fname, orient='records', lines=True)

    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None:
        args.end_idx = len(prompts)


    # make response format
    if args.experiment == 'editorials':
        if args.num_sents_per_prompt > 1:
            response_format = MultiSentenceLabelingResponse
        else:
            response_format = EditorialLabelingResponse

    
    # load the model and dependencies based on the backend
    if args.use_openai:
        load_openai_model, generate_openai_responses = load_openai_dependencies()
        tokenizer, model = load_openai_model(args.model)
        logging.info(f'loaded OpenAI model {args.model}')
    else:
        LLM, SamplingParams, AutoTokenizer, load_vllm_model, write_to_file, robust_parse_outputs, check_output_validity = load_vllm_dependencies()
        tokenizer, model = load_vllm_model(args.model)
        logging.info(f'loaded VLLM model {args.model}')

    # Create output directory if needed
    dirname = os.path.dirname(args.output_file)
    if (dirname != '') and not os.path.exists(dirname):
        os.makedirs(dirname)

    out_dirname, out_fname = os.path.split(args.output_file)
    fname, fext = os.path.splitext(out_fname)

    if args.use_openai:
        # Process all prompts at once with OpenAI
        logging.info(f'Processing {args.end_idx - args.start_idx} prompts with OpenAI')
        prompts_to_run = prompts.iloc[args.start_idx: args.end_idx]
        
        # Create output filename with indices
        output_fname = f'{out_dirname}/{fname}-labeling__experiment-{args.experiment}__model_{args.model.replace("/", "-")}__{args.start_idx}_{args.end_idx}{fext}'
        
        # Check if we already have results
        if check_existing_outputs(output_fname, args.start_idx, args.end_idx):
            logging.info(f"Skipping processing as results already exist in {output_fname}")
            exit(0)

        # Create empty file to indicate processing
        with open(output_fname, 'w') as f:
            f.write('')

        model_outputs = process_all_openai(
            model, prompts_to_run, args.model, args.batch_size, args.debug_mode, args.temp_dir, 
            args.num_sents_per_prompt,
            f"{args.experiment}__model_{args.model.replace('/', '-')}__start-idx_{args.start_idx}__end-idx_{args.end_idx}__num-sents_{args.num_sents_per_prompt}__initial-labeling",
            response_format
        )
        
        if save_outputs(output_fname, model_outputs):
            logging.info(f'-----------\nsample output: {model_outputs.iloc[0]}\n-----------\n')

    else:
        # Process in batches with VLLM
        num_batches = (args.end_idx - args.start_idx) // args.batch_size
        batch_indices = [(i * args.batch_size, min((i + 1) * args.batch_size, args.end_idx)) for i in range(num_batches)]
        random.shuffle(batch_indices)

        logging.info(f'running prompts for {args.end_idx - args.start_idx} rows')
        for start_idx, end_idx in tqdm(batch_indices):
            output_fname = f'{out_dirname}/{fname}-labeling__experiment-{args.experiment}__model_{args.model.replace("/", "-")}__{start_idx}_{end_idx}{fext}'
            
            # Check if we already have results for this batch
            if check_existing_outputs(output_fname, start_idx, end_idx):
                logging.info(f"Skipping batch {start_idx} to {end_idx} as results already exist")
                continue
                
            logging.info(f"Running prompts for batch {start_idx} to {end_idx}")
            prompts_to_run = prompts.iloc[start_idx: end_idx]
            # create an empty file to indicate that this batch is being processed
            with open(output_fname, 'w') as f:
                f.write('')
            inputted_prompts, model_outputs, outputs = process_batch_vllm(model, prompts_to_run, response_format)
            if save_outputs(output_fname, inputted_prompts, model_outputs, prompts_to_run, use_openai=False, outputs=outputs):
                logging.info(f'-----------\nsample output: {model_outputs[0]}\n-----------\n')

"""
python run_initial_labeling_prompts.py \
    --input_data_file ../data/chunks-with-problems.json.gz \
    --output_file ../data/intermediate-data/superset_labeled_data.json \
    --experiment editorials \
    --start_idx 0 \
    --end_idx 100 \
    --batch_size 5
    # --model google/gemma-2-27b-it \

python run_initial_labeling_prompts.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --input_data_file ../experiments/editorial/editorial-discourse-input-data.csv \
    --output_file ../experiments/editorial/editorial-discourse-initial-labeling.json \
    --experiment editorials \
    --start_idx 0 \
    --end_idx 100 \
    --batch_size 5

# OpenAI  
python run_initial_labeling_prompts.py \
    --model gpt-4o-mini \
    --input_data_file ../experiments/editorial/editorial-discourse-input-data.csv \
    --output_file ../experiments/editorial/editorial-discourse-initial-labeling.json \
    --experiment editorials \
    --start_idx 0 \
    --end_idx 100 \
    --batch_size 5000 \
    --use_openai \
    --debug_mode \
    --temp_dir ./debug_batches

# Multi-sentence OpenAI
python run_initial_labeling_prompts.py \
    --model gpt-4o-mini \
    --input_data_file ../experiments/editorial/editorial-discourse-input-data.csv \
    --output_file ../experiments/editorial/editorial-discourse-initial-labeling.json \
    --experiment editorials \
    --start_idx 0 \
    --end_idx 100 \
    --batch_size 5000 \
    --use_openai \
    --num_sents_per_prompt 8 \
    --debug_mode \
    --temp_dir ./debug_batches 
"""