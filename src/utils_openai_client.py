"""
Utility functions for interacting with OpenAI's batch API for efficient batch processing.
This module provides functions for loading models, creating batch files, and processing responses.
"""

import os
import json
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Type
from openai import OpenAI
from tqdm.auto import tqdm
import jsonlines
import tempfile
import time
from pathlib import Path
from datetime import datetime
import ast 
from openai.lib._parsing._completions import type_to_response_format_param
from openai.types.chat.chat_completion import Choice as OpenAIChoice

# Configure logging to suppress OpenAI client logs
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(filename)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Suppress OpenAI client logs
for logger_name in ['openai', 'httpx', 'httpcore']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)
    # Remove any existing handlers
    logger.handlers = []
    # Add a null handler to prevent propagation
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

# Global client instance
_openai_client = None

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(filename)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def robust_load_json(json_str: str) -> Dict[str, Any]:
    """
    Load JSON string robustly.
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(json_str)
        except Exception as e:
            logging.error(f"Error loading JSON string: {e}")
            return None
    


def load_model(model_name: str) -> tuple:
    """
    Initialize the OpenAI client with the specified model.
    Uses a singleton pattern to avoid reinitializing the client.
    
    Args:
        model_name: The name of the model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
        
    Returns:
        tuple: (None, client) where client is the OpenAI client instance
    """
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return None, _openai_client


def prompt_openai_model(prompt: str, model_name: str = "gpt-4o-mini", temperature: float = 0.1, response_format: Optional[type] = None) -> Any:
    """
    Send a single prompt to the model and get the response.
    
    Args:
        prompt: The prompt to send to the model
        model_name: The name of the model to use (default: "gpt-4o-mini")
        temperature: Sampling temperature (default: 0.1)
        response_format: Optional Pydantic model for structured output
        
    Returns:
        Any: The model's response (either as string or parsed structured output)
    """
    _, client = load_model(model_name)
    
    if response_format is not None:
        # Use parse method for structured output
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_format,
            temperature=temperature
        )
        return completion.choices[0].message.parsed
    else:
        # Standard text response
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content


def create_batch_files(
    prompts: List[str],
    prompt_ids: List[str],
    model_name: str,
    batch_size: int = 5000,
    temp_dir: Optional[str] = None,
    debug_mode: bool = False,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    batch_type: str = "default",
    response_format: Optional[type] = None
) -> List[Dict[str, Any]]:
    """
    Create batch files in OpenAI's format and upload them.
    
    Args:
        prompts: List of prompts to process
        model_name: Name of the model to use
        batch_size: Number of prompts per batch file
        temp_dir: Directory to store temporary files (defaults to system temp dir)
        debug_mode: If True, keeps the batch files for inspection instead of deleting them
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        batch_type: Type of batch being processed
        response_format: Pydantic model class for structured output
        
    Returns:
        List of batch file objects
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    else:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
    
    # Format prompts into OpenAI's batch format
    batched_prompts = []
    for prompt_id, prompt in zip(prompt_ids, prompts):
        request_body = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add response format if provided
        if response_format is not None:
            request_body["response_format"] = type_to_response_format_param(response_format)
        
        batched_prompts.append({
            "custom_id": prompt_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": request_body
        })
    
    # Create and upload batch files
    client = OpenAI()
    batch_files = []
    
    for i in tqdm(range(0, len(batched_prompts), batch_size), desc="Creating batch files"):
        batch = batched_prompts[i:i + batch_size]
        filename = os.path.join(temp_dir, f'batch_{i}.jsonl')
        
        # Write batch to file
        with jsonlines.open(filename, 'w') as f:
            f.write_all(batch)
        
        # Upload to OpenAI
        batch_file = client.files.create(
            file=open(filename, "rb"),
            purpose="batch"
        )
        batch_files.append(batch_file)
        
        # Clean up local file unless in debug mode
        if not debug_mode:
            os.remove(filename)
        else:
            logging.info(f"Debug mode: Keeping batch file at {filename}")
    
    return batch_files

def create_batches(
    batch_files: List[Dict[str, Any]],
    completion_window: str = "24h"
) -> List[Dict[str, Any]]:
    """
    Create batches from uploaded batch files.
    
    Args:
        batch_files: List of batch file objects from create_batch_files
        completion_window: Time window for batch completion
        
    Returns:
        List of batch objects
    """
    client = OpenAI()
    batches = []
    
    for batch_file in tqdm(batch_files, desc="Creating batches"):
        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window
        )
        batches.append(batch)
    
    return batches

def wait_for_batch_completion(batch_id: str, check_interval: int = 60) -> Dict[str, Any]:
    """
    Wait for a batch to complete and return its status.
    
    Args:
        batch_id: ID of the batch to check
        check_interval: Time in seconds between status checks
        
    Returns:
        Batch status object
    """
    client = OpenAI()
    while True:
        status = client.batches.retrieve(batch_id)
        if status.status == "completed":
            return status
        elif status.status == "failed":
            raise Exception(f"Batch {batch_id} failed: {status.error}")
        time.sleep(check_interval)

def get_batch_results(batch_id: str) -> List[Dict[str, Any]]:
    """
    Get results from a completed batch.
    
    Args:
        batch_id: ID of the completed batch
        
    Returns:
        List of results
    """
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise Exception(f"Batch {batch_id} is not completed")
    
    # Download and parse results
    file_response = client.files.content(batch.output_file_id)
    results = []
    for line in file_response.iter_lines():
        if line:
            results.append(json.loads(line))
    
    return results

def generate_responses(
    client: Any,
    prompt_ids: List[str],
    prompts: List[str],
    model_name: str,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    batch_size: int = 5000,
    debug_mode: bool = False,
    temp_dir: Optional[str] = None,
    batch_type: str = "default",
    response_format: Optional[type] = None
) -> List[str]:
    """
    Generate responses for a batch of prompts using OpenAI's batch API.
    
    Args:
        client: OpenAI client instance
        prompt_ids: List of prompt IDs to generate responses for
        prompts: List of prompts to generate responses for
        model_name: Name of the model to use
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        batch_size: Number of prompts per batch file
        debug_mode: If True, stops after creating batch files for inspection
        temp_dir: Directory to store temporary files (defaults to system temp dir)
        batch_type: Type of batch being processed
        response_format: Pydantic model class for structured output
        
    Returns:
        List of generated responses, or None if in debug mode
    """
    # Create temp directory if it doesn't exist
    if temp_dir is None:
        temp_dir = os.path.join(os.getcwd(), 'temp_batches')
    os.makedirs(temp_dir, exist_ok=True)

    # Create a unique identifier for this run
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    active_batches_file = os.path.join(temp_dir, 'active_batches.json')
    
    # Load existing active batches if any
    active_batches = {}
    logging.info(f"Checking in active batches file: {active_batches_file}")
    if os.path.exists(active_batches_file):
        try:
            with open(active_batches_file, 'r') as f:
                active_batches = json.load(f)
                # Ensure all batch IDs are strings
                active_batches = {str(k): str(v) for k, v in active_batches.items()}
                logging.info(f"Loaded {len(active_batches)} active batches from {active_batches_file}")
                logging.info(f"Active batches: {active_batches}")
        except json.JSONDecodeError:
            logging.warning("Could not parse active_batches.json, starting fresh")

    # First, check all potential batches
    batches_to_launch = []
    for i in range(0, len(prompts), batch_size):
        start_idx = i
        end_idx = min(i + batch_size, len(prompts))
        batch_id = f"{batch_type}__start-{start_idx:05d}__end-{end_idx:05d}"
        
        logging.info(f"Checking batch {batch_id}")
        logging.info(f"Batch ID in active batches: {batch_id in active_batches}")
        
        # If batch is not in active batches, we need to launch it
        if batch_id not in active_batches:
            logging.info(f"Batch {batch_id} not found in active batches, will launch")
            batches_to_launch.append((i, batch_id))
            continue
            
        # If batch is active, verify it's still valid
        try:
            logging.info(f"Batch {batch_id} is active, verifying status...")
            batch = client.batches.retrieve(str(active_batches[batch_id]))
            if batch.status == 'failed':
                logging.warning(f"Batch {batch_id} failed, will relaunch")
                batches_to_launch.append((i, batch_id))
        except Exception as e:
            logging.warning(f"Error checking batch {batch_id}: {e}, will relaunch")
            batches_to_launch.append((i, batch_id))
    
    # Launch new batches
    if batches_to_launch:
        logging.info(f"Launching {len(batches_to_launch)} new batches")
        # Create batch files for all new batches
        batch_files = create_batch_files(
            prompt_ids=prompt_ids,
            prompts=prompts,
            model_name=model_name,
            batch_size=batch_size,
            temp_dir=temp_dir,
            debug_mode=debug_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            batch_type=batch_type,
            response_format=response_format
        )
        
        if debug_mode:
            return None
            
        # Create batches from the files
        new_batches = create_batches(
            batch_files=batch_files,
            completion_window="24h"
        )
        
        # Update active batches
        for (start_idx, batch_id), batch in zip(batches_to_launch, new_batches):
            active_batches[batch_id] = str(batch.id)
            with open(active_batches_file, 'w') as f:
                json.dump(active_batches, f)
            logging.info(f"Launched new batch {batch_id}")

    # Now monitor all active batches
    all_responses = []
    while active_batches:
        completed_batches = []
        for batch_id, openai_batch_id in active_batches.items():
            try:
                batch = client.batches.retrieve(str(openai_batch_id))
                if batch.status == 'completed':
                    logging.info(f"Batch {batch_id} completed, retrieving results...")
                    # Get results from the output file
                    file_response = client.files.content(batch.output_file_id)
                    for line in file_response.iter_lines():
                        if line:
                            result = json.loads(line)
                            if "response" in result and "body" in result["response"]:
                                try:
                                    custom_id = result['custom_id']
                                    # Extract prompt index from custom_id
                                    response = robust_load_json(result['response']['body']['choices'][0]['message']['content'])
                                    all_responses.append({
                                        'custom_id': custom_id,
                                        'batch_id': batch_id,
                                        'response': response
                                    })
                                except (KeyError, json.JSONDecodeError) as e:
                                    logging.error(f"Error parsing response: {e}")
                                    all_responses.append(None)
                    completed_batches.append(batch_id)
                elif batch.status == 'failed':
                    logging.error(f"Batch {batch_id} failed: {batch.error}")
                    completed_batches.append(batch_id)
            except Exception as e:
                logging.error(f"Error checking batch {batch_id}: {e}")
                completed_batches.append(batch_id)

        # Remove completed batches
        for batch_id in completed_batches:
            del active_batches[batch_id]
            with open(active_batches_file, 'w') as f:
                pass
                # json.dump(active_batches, f)

        if active_batches:
            logging.info(f"Waiting for {len(active_batches)} batches to complete...")
            time.sleep(10)

    # Sort responses by prompt_idx to maintain order
    return all_responses


def get_openai_choice_for_scoring(
    prompt: str,
    model_name: str,
    max_tokens_to_generate: int,
    temperature: float = 0.01
) -> Optional[OpenAIChoice]:
    """
    Gets an OpenAI ChatCompletion Choice object for a given prompt,
    requesting logprobs. Intended for scoring/analysis purposes.
    max_tokens_to_generate is crucial: if 0, logprobs from OpenAI are usually None for content.
    """
    _, client = load_model(model_name) # Uses global _openai_client
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens_to_generate,
            logprobs=True # Always request logprobs
        )
        if response.choices:
            return response.choices[0]
        else:
            logging.warning(f"OpenAI call for scoring returned no choices for prompt: {prompt[:100]}...")
            return None
    except Exception as e:
        logging.error(f"Error calling OpenAI API for scoring prompt {prompt[:100]}...: {e}")
        return None