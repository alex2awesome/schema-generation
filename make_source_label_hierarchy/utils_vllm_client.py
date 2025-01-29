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
            enforce_eager=True,
            max_model_len=60_000
        )
    return _tokenizer, _model


def run_vllm_batch(prompts, model_name, sampling_params=None):
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.1, max_tokens=4096)

    tokenizer, model = load_model(model_name)
    prompt_dicts = list(map(lambda x: [
        {
            "role": "system",
            "content": "You are an experienced journalist.",
        },

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
    text_results = list(map(lambda x: x.outputs[0].text, results))
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