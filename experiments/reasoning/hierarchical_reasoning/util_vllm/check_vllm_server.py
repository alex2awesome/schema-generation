#!/usr/bin/env python3
"""
Check if VLLM server is running and test a simple query.

Usage:
    python check_vllm_server.py
    python check_vllm_server.py --port 8001
"""

import argparse
import requests
import json
import time


def check_server_health(base_url: str) -> bool:
    """Check if the VLLM server is healthy."""
    try:
        # VLLM health endpoint is at /health, not /v1/health
        health_url = base_url.replace("/v1", "") + "/health"
        response = requests.get(health_url, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def test_simple_query(base_url: str, model_name: str) -> bool:
    """Test a simple query to the VLLM server."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hello! Please respond with 'Server is working correctly.'"}],
        "temperature": 0.1,
        "max_tokens": 50,
        "stream": False
    }
    
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        response_text = result["choices"][0]["message"]["content"]
        
        print(f"✅ Test query successful!")
        print(f"Response: {response_text}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Test query failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error parsing response: {e}")
        return False


def get_model_list(base_url: str) -> bool:
    """Get list of available models."""
    try:
        response = requests.get(f"{base_url}/models", timeout=10)
        response.raise_for_status()
        
        models = response.json()
        print(f"📋 Available models:")
        for model in models.get("data", []):
            print(f"  - {model.get('id', 'Unknown')}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to get model list: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check VLLM server status")
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port where VLLM server is running"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost",
        help="Host where VLLM server is running"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Model name to test (if not provided, will use first available model)"
    )
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}/v1"
    
    print(f"🔍 Checking VLLM server at {base_url}")
    print("-" * 50)
    
    # Check server health
    print("1. Checking server health...")
    if check_server_health(base_url):
        print("✅ Server is healthy!")
    else:
        print("❌ Server is not responding or unhealthy")
        print("   Make sure the VLLM server is running:")
        print(f"   python start_vllm_server.py --port {args.port}")
        return
    
    # Get model list
    print("\n2. Getting available models...")
    if not get_model_list(base_url):
        return
    
    # Test simple query
    print("\n3. Testing simple query...")
    if args.model:
        model_name = args.model
    else:
        # Try to get first model from the list
        try:
            response = requests.get(f"{base_url}/models", timeout=10)
            models = response.json()
            if models.get("data"):
                model_name = models["data"][0]["id"]
            else:
                print("❌ No models available")
                return
        except:
            print("❌ Could not determine model name")
            return
    
    print(f"Using model: {model_name}")
    test_simple_query(base_url, model_name)
    
    print("\n" + "=" * 50)
    print("✅ VLLM server check complete!")
    print(f"Server URL: {base_url}")
    print("You can now use this server with your hierarchical reasoning code.")


if __name__ == "__main__":
    main() 