import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import unicodedata

import os
import json
import torch
import ast
import logging
import re 

from vllm import LLM,  SamplingParams
from transformers import AutoTokenizer

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

_model = None
_tokenizer = None
def load_model(model_name: str):
    global _model, _tokenizer
    if _model is None:
        torch.cuda.memory_summary(device=None, abbreviated=False)
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = LLM(
            model_name,
            dtype=torch.float16,
            tensor_parallel_size=torch.cuda.device_count(),
            # enforce_eager=True,
            max_model_len=10_000,
            gpu_memory_utilization=0.95,
        )
    return _tokenizer, _model


def run_vllm_batch(prompts, model_name, sampling_params=None, include_system_prompt=True):
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.1, max_tokens=4096)

    tokenizer, model = load_model(model_name)
    if include_system_prompt:
        prompt_dicts = list(map(lambda x: [
            {
                "role": "system",
                "content": "You are an experienced analyst.",
            },

            {
                "role": "user",
                "content": x,
            },
        ], prompts))
    else:
        prompt_dicts = list(map(lambda x: [
            {
                "role": "user",
                "content": x,
            },
        ], prompts))
    formatted_prompts = list(map(lambda x: 
        tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True), 
        prompt_dicts
    ))
    results = model.generate(formatted_prompts, sampling_params=sampling_params)
    sorted_results = sorted(results, key=lambda x: int(x.request_id))
    text_results = list(map(lambda x: x.outputs[0].text, sorted_results))
    return text_results


def match_batched_vllm_results_to_prompts(all_results, all_batched_inputs):
    """
    Match prompts to VLLM results when each prompt is a batch of multiple queries.
    """
    all_merged = []
    for r, b in zip(all_results, all_batched_inputs):
        r = r.split('\n')
        if len(r) == len(b):
            b['label'] = r 
            all_merged.append(b)

    full_data_df = pd.concat(all_merged)
    full_data_df['label'] = full_data_df['label'].str.replace(r'\d\.', '', regex=True).str.strip()
    return full_data_df


def write_to_file(fname, indices=None, outputs=None, index_output_chunk=None):
    with open(fname, 'ab') as file:
        if index_output_chunk is None:
            index_output_chunk = zip(indices, outputs)

        for index, output in index_output_chunk:
            if isinstance(output, str):
                response = output
            else:
                response = output.outputs[0].text
            response = unicodedata.normalize('NFKC', response)
            if response and index:
                output = {}
                output['index'] = str(index)
                output['response'] = robust_parse_outputs(response)
                file.write(json.dumps(output).encode('utf-8'))
                file.write(b'\n')


def extract_list_brackets_from_text(text):
    pattern = r'\[.*\]'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


import json, ast 
def robust_parse_outputs(output):
    output = extract_list_brackets_from_text(output)
    if output is None:
        return None
    try:
        return _robust_parse_outputs(output)
    except:
        try:
            return _robust_parse_outputs(output.replace('\n', ' '))
        except:
            return None


def _robust_parse_outputs(output):
    try:
        return ast.literal_eval(output)
    except:
        try:
            return json.loads(output)
        except:
            return None


def check_output_validity(output):
    """
    Check if the outputs are valid.
    """
    if robust_parse_outputs(output) is None:
        return False
    return True