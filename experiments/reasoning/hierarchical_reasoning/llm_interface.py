"""
LLM Interface for Hierarchical Reasoning

This module provides a flexible interface for interacting with different LLM frameworks,
allowing for easy extension to new providers. It currently supports OpenAI, VLLM, and TogetherAI.

How to Add a New LLM Framework:
-------------------------------

To add support for a new LLM provider, follow these steps:

1.  **Create a New Interface Class**:
    - Your new class should inherit from `BaseLLMInterface`.
    - This ensures your class conforms to the required structure.

2.  **Implement the `__call__` Method**:
    - Your class must implement the `__call__` method with the following signature:
      ```python
      def __call__(
          self,
          prompt: str,
          model_name: str,
          temperature: float,
          response_format: Optional[Type[BaseModel]] = None,
          **kwargs
      ) -> Union[str, BaseModel]:
      ```

3.  **Handle Output Formats**:
    - The `__call__` method's behavior should change based on the `response_format` argument:
      - **If `response_format` is `None`**: The method must return the raw text response from the LLM as a `str`.
      - **If `response_format` is a Pydantic `BaseModel` class**: The method must return an instance of that Pydantic model.
        - If the underlying API supports structured output (e.g., JSON mode), use it to get a structured response.
        - If not, you can use the helper functions `format_prompt_for_structured_output` to request JSON in the prompt and `parse_response_for_structured_output` to parse the text response into the Pydantic model.

4.  **Update the Factory Function**:
    - Add your new class to the `get_llm_interface` factory function with a unique key for the `llm_framework` argument.

By following this pattern, the new LLM interface will be seamlessly integrated into the
existing reasoning and training scripts.
"""

from typing import Any, Optional, Type, Union
from pydantic import BaseModel, Field
import os
import subprocess
import time
import signal
import atexit
from pathlib import Path
import sys
import together
import requests
import json
import logging
from addict import Dict as AttrDict
import ast 
import re

# Configure logging for this module
logger = logging.getLogger(__name__)

here = Path(__file__).parent
sys.path.append(str(here.parent.parent.parent / 'src'))

# Import functions from our existing scripts
try:
    from experiments.reasoning.hierarchical_reasoning.util_vllm.start_vllm_server import start_vllm_server_background
    from experiments.reasoning.hierarchical_reasoning.util_vllm.check_vllm_server import check_server_health, test_simple_query
except ImportError as e:
    logger.warning(f"Could not import VLLM utilities: {e}")
    start_vllm_server_background = None
    check_server_health = None
    test_simple_query = None

TOGETHER_SUPPORTED_STRUCTURED_OUTPUT_MODELS = [
    'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
    'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
    'meta-llama/Llama-3.2-3B-Instruct-Turbo',
    'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
    'meta-llama/Llama-4-Scout-17B-16E-Instruct',
    'deepseek-ai/DeepSeek-V3',
    'deepseek-ai/DeepSeek-R1',
    'Qwen/Qwen3-235B-A22B-fp8-tput',
    'Qwen/Qwen2.5-VL-72B-Instruct',
    'arcee_ai/arcee-spotlight',
    'arcee-ai/arcee-blitz',
]

def robust_json_loads(json_str: str) -> Any:
    """Robustly load a JSON string by trying json.loads and then ast.literal_eval."""
    if not json_str:
        return None
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # Fallback for Python-style literals like None, True, False
            return ast.literal_eval(json_str)
        except (ValueError, SyntaxError, MemoryError, TypeError):
            # If it's not valid JSON or a valid Python literal, return as is.
            return json_str



def format_prompt_for_structured_output(response_format: Type[BaseModel]) -> str:
    """
    Format a prompt for structured output.
    This is a hack to get structured output from LLMs that don't support it natively.
    """

    # Instead of including the full schema, just specify the expected format
    format_prompt = "\n\nPlease respond with a JSON object containing the following fields:\n"
    for field_name, field_info in response_format.__annotations__.items():
        if hasattr(response_format, 'model_fields') and field_name in response_format.model_fields:
            field = response_format.model_fields[field_name]
            description = field.description if hasattr(field, 'description') else field_name
            format_prompt += f"- {field_name}: {description}\n"
    format_prompt += "\n\nPlease be exact in your response, do not miss any fields or include any other text."
    return format_prompt


def parse_response_for_structured_output(response_content: str, response_format: Type[BaseModel]) -> BaseModel:
    json_objects = re.findall(r'\{[^{}]*\}', response_content)
    
    # Try each JSON object until we find one that matches our schema
    for json_str in json_objects:
        try:
            parsed_json = robust_json_loads(json_str)
            # Check if this JSON has the expected structure
            if all(key in parsed_json for key in response_format.__annotations__):
                return response_format(**parsed_json)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    logger.warning(f"Could not parse structured output from any JSON in response")
    logger.warning(f"Expected model: {response_format.__name__}")
    logger.warning(f"Raw response: {response_content}")
    
    # Final fallback: return the text response
    logger.warning("Falling back to text response")
    return response_content
    
def get_llm_interface(
    llm_framework: str = "openai",  # "vllm"
    model_name: str = "gpt-4o-mini", # "meta-llama/Meta-Llama-3.1-8B-Instruct"
    temperature: float = 0.1, 
    vllm_base_url: str = "http://localhost:8000/v1", # only needed for vllm
    vllm_gpus: str = "2,3", # only needed for vllm
    vllm_auto_start: bool = True, # only needed for vllm
    max_tokens: int = 2048,
    verbose: bool = True,
) -> Any:
    """
    Factory function to get the appropriate LLM interface.
    
    Args:
        llm_framework: The LLM framework to use ("openai", "vllm", etc.)
        model_name: The model name to use
        temperature: Sampling temperature
        base_url: Base URL for VLLM server (only used for vllm framework)
        gpus: GPUs to use for VLLM server (only used for vllm framework)
        auto_start_server: Whether to automatically start VLLM server (only used for vllm framework)
    
    Returns:
        An LLM interface object with a call method
    """
    if llm_framework.lower() == "openai":
        return OpenAIInterface(model_name=model_name, temperature=temperature)
    elif llm_framework.lower() == "together":
        if model_name not in TOGETHER_SUPPORTED_STRUCTURED_OUTPUT_MODELS:
            logger.warning(f"Model will not support structured output: {model_name}. If you want Structured Output, consider using one of these models:\n{chr(10).join(TOGETHER_SUPPORTED_STRUCTURED_OUTPUT_MODELS)}")
        return TogetherAIInterface(model_name=model_name, temperature=temperature, structured_outputs_supported=model_name in TOGETHER_SUPPORTED_STRUCTURED_OUTPUT_MODELS)
    elif llm_framework.lower() == "vllm":
        return VLLMInterface(
            model_name=model_name, 
            temperature=temperature, 
            base_url=vllm_base_url,
            gpus=vllm_gpus,
            auto_start_server=vllm_auto_start,
            max_tokens=max_tokens,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unsupported LLM framework: {llm_framework}")


class BaseLLMInterface:
    """Base class for LLM interfaces"""
    def __call__(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.1,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> Union[str, BaseModel]:
        """
        Call the LLM with the given prompt and parameters.
        
        Args:
            prompt: The prompt to send to the LLM
            model_name: The name of the model to use
            temperature: Sampling temperature
            response_format: Optional Pydantic model for structured output
            **kwargs: Additional framework-specific parameters
        
        Returns:
            Either a string response or a Pydantic model instance
        """
        raise NotImplementedError("Subclasses must implement __call__")


class TogetherAIInterface(BaseLLMInterface):
    """Interface for TogetherAI's API"""

    def __init__(
            self, 
            model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", 
            temperature: float = 0.1, 
            structured_outputs_supported: bool = True,
            **kwargs
    ):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.structured_outputs_supported = structured_outputs_supported
        self.client = together.Together()

    def __call__(
        self,
        prompt: str,
        model_name: str = None,
        temperature: float = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> Union[str, BaseModel]:
        """
        Call TogetherAI's API with the given prompt and parameters.
        """
        expects_response = response_format is not None
        if expects_response:
            if self.structured_outputs_supported:
                response_format = {
                    "type": "json_object",
                    "schema": response_format.model_json_schema(),
                }
            else:
                format_prompt = format_prompt_for_structured_output(response_format)
                prompt = prompt + format_prompt

        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model=model_name or self.model_name,
            response_format=response_format if self.structured_outputs_supported else None,
        )

        response_content = response.choices[0].message.content
        if expects_response:
            if self.structured_outputs_supported:
                return AttrDict(robust_json_loads(response_content))
            else:
                return parse_response_for_structured_output(response_content, response_format)
        else:
            return response_content
        

class OpenAIInterface(BaseLLMInterface):
    """Interface for OpenAI's API"""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature

    def __call__(
        self,
        prompt: str,
        model_name: str = None,
        temperature: float = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> Union[str, BaseModel]:
        """
        Call OpenAI's API with the given prompt and parameters.
        
        Args:
            prompt: The prompt to send to the LLM
            model_name: The name of the model to use
            temperature: Sampling temperature
            response_format: Optional Pydantic model for structured output
            **kwargs: Additional OpenAI-specific parameters
        
        Returns:
            Either a string response or a Pydantic model instance
        """
        from utils_openai_client import prompt_openai_model as openai_prompt
        try:
            response = openai_prompt(
                model_name=model_name or self.model_name,
                prompt=prompt,
                temperature=temperature or self.temperature,
                response_format=response_format,
                **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise


class VLLMInterface(BaseLLMInterface):
    """Interface for VLLM server with automatic server management"""

    def __init__(
            self, 
            model_name: str = "meta-llama/Llama-3.1-8B-Instruct", 
            temperature: float = 0.1, 
            base_url: str = "http://localhost:8000/v1",
            max_tokens: int = 2048,
            gpus: str = "2,3",
            auto_start_server: bool = True,
            verbose: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.gpus = gpus
        self.auto_start_server = auto_start_server
        self.verbose = verbose
        self.server_process = None
        
        # Extract port from base_url
        import re
        port_match = re.search(r':(\d+)', base_url)
        self.port = int(port_match.group(1)) if port_match else 8000
        
        # Register cleanup on exit and signal handlers
        atexit.register(self._cleanup_server)
        self._register_signal_handlers()
        
        # Auto-start server if enabled
        if self.auto_start_server:
            try:
                self._ensure_server_running()
            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt detected during server startup...")
                self._cleanup_server()
                raise

    def _register_signal_handlers(self):
        """Register signal handlers for proper cleanup on interrupts."""
        try:
            import signal
            # Handle common interrupt signals
            signal.signal(signal.SIGINT, self._signal_handler)  # Ctrl+C
            signal.signal(signal.SIGTERM, self._signal_handler)  # Termination
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, self._signal_handler)  # Hangup (Unix only)
        except Exception as e:
            logger.warning(f"Could not register signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals by cleaning up the server."""
        logger.warning(f"Received signal {signum}, cleaning up VLLM server...")
        self._cleanup_server()
        # Re-raise the KeyboardInterrupt to maintain normal interrupt behavior
        if signum == signal.SIGINT:
            raise KeyboardInterrupt()
        else:
            exit(1)

    def _is_server_running(self) -> bool:
        """Check if the VLLM server is running and healthy."""
        if check_server_health is None:
            # Fallback implementation if import failed
            import requests
            try:
                health_url = f"http://localhost:{self.port}/health"
                response = requests.get(health_url, timeout=5)
                return response.status_code == 200
            except requests.exceptions.RequestException:
                return False
        
        # Use the imported function
        return check_server_health(self.base_url)

    def _start_server(self) -> bool:
        """Start VLLM server in background."""
        if self._is_server_running():
            logger.info(f"VLLM server already running on port {self.port}")
            return True
        
        if start_vllm_server_background is None:
            logger.error("start_vllm_server_background function not available")
            return False
            
        # Use the imported function
        log_file_path = here / f"vllm_server_{self.port}.log"
        self.server_process = start_vllm_server_background(
            model_name=self.model_name,
            port=self.port,
            gpus=self.gpus,
            max_model_len=None,
            trust_remote_code=True,
            log_file=str(log_file_path),
            verbose=self.verbose
        )
        
        if self.server_process is None:
            return False
        
        # Save PID for cleanup
        pid_file_path = here / f"vllm_server_{self.port}.pid"
        with open(pid_file_path, 'w') as pid_file:
            pid_file.write(str(self.server_process.pid))
        
        return True

    def _wait_for_server(self, timeout: int = 60 * 60, test_query: bool = False) -> bool:
        """Wait for server to be ready."""
        logger.info(f"Waiting for server to be ready (timeout: {timeout}s)...")
        
        start_time = time.time()
        try:
            while time.time() - start_time < timeout:
                if self._is_server_running():
                    logger.info("Server is ready!")
                    # Optionally test with a simple query
                    if test_query and test_simple_query is not None:
                        logger.info("Testing server with simple query...")
                        test_simple_query(self.base_url, self.model_name)
                    return True
                time.sleep(2)
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt during server wait...")
            raise
        
        logger.error(f"Server failed to start within {timeout} seconds")
        return False

    def _ensure_server_running(self) -> bool:
        """Ensure VLLM server is running, start it if necessary."""
        if self._is_server_running():
            return True
            
        if not self._start_server():
            return False
            
        return self._wait_for_server()

    def _cleanup_server(self):
        """Clean up server process on exit."""
        try:
            # Check if we have a server process to clean up
            if hasattr(self, 'server_process') and self.server_process and self.server_process.poll() is None:
                logger.info(f"Cleaning up VLLM server (PID: {self.server_process.pid})...")
                
                # Kill the process group to ensure all child processes are terminated
                if hasattr(os, 'killpg'):
                    try:
                        os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        # Process already dead
                        pass
                else:
                    self.server_process.terminate()
                
                # Wait a bit for graceful shutdown
                try:
                    self.server_process.wait(timeout=5)
                    logger.info("Server shutdown gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("Server didn't shutdown gracefully, force killing...")
                    # Force kill if it doesn't shut down gracefully
                    if hasattr(os, 'killpg'):
                        try:
                            os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        self.server_process.kill()
                    logger.info("Server force killed")
                        
            # Also try to find and kill any orphaned processes by PID file
            elif hasattr(self, 'port'):
                pid_file_path = here / f"vllm_server_{self.port}.pid"
                if pid_file_path.exists():
                    try:
                        with open(pid_file_path, 'r') as f:
                            pid = int(f.read().strip())
                        logger.info(f"Found orphaned VLLM server PID {pid}, cleaning up...")
                        if hasattr(os, 'killpg'):
                            try:
                                os.killpg(os.getpgid(pid), signal.SIGTERM)
                                time.sleep(2)
                                os.killpg(os.getpgid(pid), signal.SIGKILL)  # Ensure it's dead
                            except ProcessLookupError:
                                pass
                        else:
                            os.kill(pid, signal.SIGTERM)
                            time.sleep(2)
                            try:
                                os.kill(pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        logger.info("Orphaned server cleanup complete")
                    except (ValueError, OSError, ProcessLookupError):
                        pass
                        
        except Exception as e:
            logger.warning(f"Error during server cleanup: {e}")
        
        finally:
            # Clean up PID file regardless
            try:
                if hasattr(self, 'port'):
                    pid_file_path = here / f"vllm_server_{self.port}.pid"
                    if pid_file_path.exists():
                        pid_file_path.unlink()
            except Exception:
                pass

    def __call__(
        self,
        prompt: str,
        model_name: str = None,
        temperature: float = None,
        response_format: Optional[Type[BaseModel]] = None,
        add_response_format_to_prompt: bool = True,
        **kwargs
    ) -> Union[str, BaseModel]:
        """
        Call VLLM server with the given prompt and parameters.
        
        Args:
            prompt: The prompt to send to the LLM
            model_name: The name of the model to use
            temperature: Sampling temperature
            response_format: Optional Pydantic model for structured output
            **kwargs: Additional VLLM-specific parameters
        
        Returns:
            Either a string response or a Pydantic model instance
        """        
        # Prepare the request payload
        # If structured output is requested, modify the prompt to include schema
        final_prompt = prompt
        if response_format is not None and add_response_format_to_prompt:
            format_prompt = format_prompt_for_structured_output(response_format)
            final_prompt = prompt + format_prompt
        
        payload = {
            "model": model_name or self.model_name,
            "messages": [{"role": "user", "content": final_prompt}],
            "temperature": temperature or self.temperature,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False
        }
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ["max_tokens"]:  # Skip already handled params
                payload[key] = value
        
        try:
            # Make request to VLLM server
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=300  # 5 minute timeout
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            text_response = result["choices"][0]["message"]["content"]
            
            # Handle structured output if requested
            if response_format is not None:
                return parse_response_for_structured_output(text_response, response_format)
            else:
                return text_response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling VLLM server: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing VLLM response: {e}")
            raise

# Default interface instance
default_llm = get_llm_interface(os.getenv("LLM_FRAMEWORK", "openai")) 



"""
```python
from llm_interface import get_llm_interface
llm = get_llm_interface(
    llm_framework="vllm",
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", 
    temperature=0.1, 
    base_url="http://localhost:8000/v1", 
    gpus="2,3", 
    auto_start_server=True,
    max_tokens=2048,
)
response = llm("What is the capital of France?")
print(response)

# test with Structured Output
from pydantic import BaseModel, Field
class Response(BaseModel):
    city: str = Field(description="The capital city")
    country: str = Field(description="The country")

response = llm("What is the capital of France? Answer in JSON format, with city and country keys.", response_format=Response)
print(response)


# test with Structured Output
class CityInfo(BaseModel):
    city: str = Field(description="The capital city name")
    country: str = Field(description="The country name")
    population: int = Field(description="Approximate population in millions")

response = llm(
    "What is the capital of France?", 
    response_format=CityInfo
)

```
"""

