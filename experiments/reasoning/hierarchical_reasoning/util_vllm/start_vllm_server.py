#!/usr/bin/env python3
"""
Start VLLM server on 2 GPUs for hierarchical reasoning experiments.

Usage:
    python start_vllm_server.py --model meta-llama/Meta-Llama-3.1-8B-Instruct
    python start_vllm_server.py --model Qwen/QwQ-32B-Preview --port 8001
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import threading
import queue
import shutil
import psutil

# Set up logging
def setup_logging(log_file, verbose=True):
    """Set up the logger for the server."""
    logger = logging.getLogger('vllm_server')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear any existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    if not verbose:
        logger.addHandler(logging.NullHandler())
        return logger
    
    # Console handler with custom format
    console_handler = logging.StreamHandler(sys.stdout) # Explicitly use stdout
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')  # Simpler format for console
    console_handler.setFormatter(console_format)
    
    # File handler with detailed format
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def read_output(pipe, logger, queue_for_check=None):
    """Read output from a pipe, log it, and optionally put it on a queue."""
    for line in iter(pipe.readline, ''):
        line_strip = line.strip()
        if line_strip:
            logger.info(line_strip) # Log to console and file via handlers
            if queue_for_check:
                queue_for_check.put(line_strip)
    pipe.close()

def _start_and_log_process(cmd, env, logger, log_file_path):
    """
    Starts a process, sets its priority, and starts a background thread to log its output.
    Returns the process object.
    """
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    # Boost process priority using psutil
    if os.environ.get('VLLM_SET_PRIORITY', '0') == '1':
        try:
            p = psutil.Process(process.pid)
            p.nice(-5)
            logger.info(f"Successfully set CPU nice level to -5 for process {p.pid}")
            if hasattr(p, 'ionice'):
                p.ionice(psutil.IOPRIO_CLASS_BE, value=0)
                logger.info(f"Successfully set I/O priority for process {p.pid}")
        except psutil.AccessDenied as e:
            logger.warning(f"Permission denied to boost process priority: {e}. Continuing.")
        except psutil.NoSuchProcess:
            logger.warning(f"Process {process.pid} ended before priority could be set.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while setting process priority: {e}")

    # Start a thread to read and log the output
    log_thread = threading.Thread(
        target=read_output,
        args=(process.stdout, logger),
        daemon=True
    )
    log_thread.start()

    return process

def log_system_info(logger):
    """Log versions of key libraries."""
    try:
        import torch
        import vllm
        import transformers
        logger.info("="*20 + " System Info " + "="*20)
        logger.info(f"Python version: {sys.version.split()[0]}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"VLLM version: {vllm.__version__}")
        logger.info(f"Transformers version: {transformers.__version__}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Detected {torch.cuda.device_count()} GPUs")
        logger.info("="*53)
    except ImportError as e:
        logger.warning(f"Could not import a library to log system info: {e}")

def start_vllm_server(model_name: str, port: int = 8000, gpus: str = "0,1", 
                     max_model_len: int = None, trust_remote_code: bool = True,
                     cache_dir: str = None, verbose: bool = True):
    """Start VLLM server with specified configuration (foreground)."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create a log file for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"vllm_server_{timestamp}.log"
    
    # Set up logging
    logger = setup_logging(log_file, verbose=verbose)
    
    # Log system info
    log_system_info(logger)

    # Set environment variables for the subprocess
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpus
    env['VLLM_LOG_LEVEL'] = 'DEBUG'  # More verbose VLLM logging

    # --- Cache Directory Logic ---
    cache_path_to_set = None
    if cache_dir:
        # 1. User-specified --cache-dir takes precedence
        cache_path_to_set = Path(cache_dir)
        logger.info(f"Using user-specified cache directory via --cache-dir: {cache_path_to_set}")
    elif 'HF_HOME' not in env:
        # 2. HF_HOME is not set, so create a temporary one.
        if 'SLURM_TMPDIR' in env:
            cache_path_to_set = Path(env['SLURM_TMPDIR']) / 'vllm_cache'
            logger.info(f"HF_HOME not set. Using SLURM temporary directory for cache: {cache_path_to_set}")
        else:
            cache_path_to_set = Path(f"/tmp/vllm_cache_{os.getuid()}")
            logger.info(f"HF_HOME not set. Using fallback temporary directory for cache: {cache_path_to_set}")

    if cache_path_to_set:
        cache_path_to_set.mkdir(parents=True, exist_ok=True)
        env['HF_HOME'] = str(cache_path_to_set)
        env['HUGGING_FACE_HUB_CACHE'] = str(cache_path_to_set / 'huggingface' / 'hub')
    else:
        # 3. HF_HOME is already set and not overridden by --cache-dir
        logger.info(f"Using existing cache directory from HF_HOME: {env['HF_HOME']}")
    
    # Build the command
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--host", "0.0.0.0",
        "--tensor-parallel-size", str(len(gpus.split(','))),  # Number of GPUs
        "--disable-log-requests",  # Reduce noise in logs
    ]
    
    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])
    
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    
    logger.info(f"Starting VLLM server with command: {' '.join(cmd)}")
    logger.info(f"Using GPUs: {gpus}")
    logger.info(f"Server will be available at http://localhost:{port}")
    logger.info(f"OpenAI-compatible endpoint: http://localhost:{port}/v1")
    logger.info(f"Log file: {log_file}")
    
    process = None
    try:
        # Start the server and get the process object
        process = _start_and_log_process(cmd, env, logger, log_file)
        # Wait for the process to complete (since this is a foreground function)
        process.wait()
        logger.info(f"Server process {process.pid} finished.")
            
    except KeyboardInterrupt:
        logger.info("\nShutting down VLLM server...")
        if process:
            process.terminate()
    except Exception as e:
        logger.error(f"Unexpected error in start_vllm_server: {e}", exc_info=True)
        if process:
            process.terminate()
        sys.exit(1)

def start_vllm_server_background(model_name: str, port: int = 8000, gpus: str = "0,1", 
                                max_model_len: int = None, trust_remote_code: bool = True,
                                log_file: str = None, cache_dir: str = None, verbose: bool = True):
    """Start VLLM server in background and return the process."""
    
    # Set up logging and environment
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_file or log_dir / f"vllm_server_bg_{port}.log"
    logger = setup_logging(log_file_path, verbose=verbose)
    log_system_info(logger)
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpus
    env['VLLM_LOG_LEVEL'] = 'DEBUG'

    # --- Cache Directory Logic ---
    cache_path_to_set = None
    if cache_dir:
        # 1. User-specified --cache-dir takes precedence
        cache_path_to_set = Path(cache_dir)
        logger.info(f"Using user-specified cache directory via --cache-dir: {cache_path_to_set}")
    elif 'HF_HOME' not in env:
        # 2. HF_HOME is not set, so create a temporary one.
        if 'SLURM_TMPDIR' in env:
            cache_path_to_set = Path(env['SLURM_TMPDIR']) / 'vllm_cache'
            logger.info(f"HF_HOME not set. Using SLURM temporary directory for cache: {cache_path_to_set}")
        else:
            cache_path_to_set = Path(f"/tmp/vllm_cache_{os.getuid()}")
            logger.info(f"HF_HOME not set. Using fallback temporary directory for cache: {cache_path_to_set}")

    if cache_path_to_set:
        cache_path_to_set.mkdir(parents=True, exist_ok=True)
        env['HF_HOME'] = str(cache_path_to_set)
        env['HUGGING_FACE_HUB_CACHE'] = str(cache_path_to_set / 'huggingface' / 'hub')
    else:
        # 3. HF_HOME is already set and not overridden by --cache-dir
        logger.info(f"Using existing cache directory from HF_HOME: {env.get('HF_HOME', 'Not Set')}")
    
    # Build the command
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--host", "0.0.0.0",
        "--tensor-parallel-size", str(len(gpus.split(','))),  # Number of GPUs
        "--disable-log-requests",  # Reduce noise in logs
    ]
    
    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])
    
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    
    logger.info(f"🚀 Starting VLLM server in background with model {model_name} on GPUs {gpus}...")
    logger.info(f"📝 Server logs will be streamed to console and saved to: {log_file_path}")
    
    try:
        # Start the server and return the process object immediately
        process = _start_and_log_process(cmd, env, logger, log_file_path)
        logger.info(f"✅ Server process {process.pid} launched in background.")
        return process
        
    except Exception as e:
        logger.error(f"❌ Failed to start VLLM server in background: {e}", exc_info=True)
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Start VLLM server for hierarchical reasoning",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model name to load in VLLM server"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port for VLLM server"
    )
    parser.add_argument(
        "--gpus", 
        type=str, 
        default="0,1",
        help="Comma-separated list of GPU IDs to use"
    )
    parser.add_argument(
        "--max-model-len", 
        type=int, 
        default=None,
        help="Maximum model length (optional)"
    )
    parser.add_argument(
        "--trust-remote-code", 
        action="store_true",
        default=True,
        help="Trust remote code for model loading"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Path to use for the model cache. Overrides HF_HOME.\n"
             "If not set, defaults to the HF_HOME environment variable.\n"
             "If HF_HOME is also not set, it will try to use $SLURM_TMPDIR, then /tmp."
    )
    parser.add_argument(
        "--start-vllm-verbose",
        action="store_true",
        help="Enable detailed server logging to console and file. Default is quiet."
    )
    
    args = parser.parse_args()
    
    start_vllm_server(
        model_name=args.model,
        port=args.port,
        gpus=args.gpus,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        verbose=args.start_vllm_verbose
    )

if __name__ == "__main__":
    main() 