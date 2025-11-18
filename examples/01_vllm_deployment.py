"""
Example: Deploying LLMs with vLLM

This example shows how to:
1. Start vLLM with a model
2. Connect Modelium to it
3. Run inference through Modelium

Prerequisites:
- vLLM installed: pip install vllm
- Or vLLM Docker container running
"""

import subprocess
import time
import requests
import json

def start_vllm_server():
    """
    Start vLLM server with GPT-2.
    
    In production, you'd run this separately:
        docker run --gpus all -p 8001:8000 vllm/vllm-openai:latest --model gpt2
    """
    print("🚀 Starting vLLM server...")
    print("   Model: gpt2")
    print("   Port: 8001")
    print()
    
    # Start vLLM in background
    # Note: In production, run this as a separate service
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "gpt2",
        "--host", "0.0.0.0",
        "--port", "8001",
        "--dtype", "auto",
    ]
    
    # This would start the server (commented out for example)
    # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    print("⚠️  In this example, we assume vLLM is already running.")
    print("   Start it with:")
    print("   docker run --gpus all -p 8001:8000 vllm/vllm-openai:latest --model gpt2")
    print()


def check_vllm_health():
    """Check if vLLM is running."""
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def test_vllm_directly():
    """Test vLLM's OpenAI-compatible API directly."""
    print("🧪 Testing vLLM directly...")
    
    response = requests.post(
        "http://localhost:8001/v1/completions",
        json={
            "model": "gpt2",
            "prompt": "Once upon a time",
            "max_tokens": 50,
            "temperature": 0.7,
        }
    )
    
    result = response.json()
    print(f"   Generated: {result['choices'][0]['text'][:100]}...")
    print("   ✅ vLLM working!")
    print()
    return result


def test_via_modelium():
    """Test inference through Modelium's unified API."""
    print("🧪 Testing via Modelium...")
    
    # First, check if model is registered
    response = requests.get("http://localhost:8000/models")
    models = response.json()
    print(f"   Registered models: {[m['name'] for m in models['models']]}")
    
    # Run inference
    response = requests.post(
        "http://localhost:8000/predict/gpt2",
        json={
            "prompt": "The future of AI is",
            "organizationId": "example-org",
            "max_tokens": 50,
            "temperature": 0.7,
        }
    )
    
    result = response.json()
    print(f"   Generated: {result['choices'][0]['text'][:100]}...")
    print("   ✅ Modelium routing working!")
    print()


def main():
    """Main example workflow."""
    print("=" * 70)
    print("vLLM Deployment Example")
    print("=" * 70)
    print()
    
    # Step 1: Check vLLM
    print("Step 1: Verify vLLM is running")
    if not check_vllm_health():
        print("❌ vLLM not running!")
        print()
        print("Please start vLLM first:")
        print("  docker run --gpus all -p 8001:8000 vllm/vllm-openai:latest --model gpt2")
        print()
        return
    
    print("✅ vLLM is running")
    print()
    
    # Step 2: Test vLLM directly
    print("Step 2: Test vLLM API")
    test_vllm_directly()
    
    # Step 3: Test via Modelium
    print("Step 3: Test Modelium routing")
    print("   (Make sure Modelium server is running: python -m modelium.cli serve)")
    
    try:
        test_via_modelium()
    except requests.exceptions.ConnectionError:
        print("❌ Modelium not running!")
        print("   Start it with: python -m modelium.cli serve")
        print()
        return
    
    # Success!
    print("=" * 70)
    print("✅ All tests passed!")
    print()
    print("You can now:")
    print("  - Drop more models in models/incoming/")
    print("  - Access them via http://localhost:8000/predict/{model-name}")
    print("  - Monitor with curl http://localhost:8000/status")
    print("=" * 70)


if __name__ == "__main__":
    main()

