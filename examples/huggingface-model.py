#!/usr/bin/env python3
"""
Example: Deploy HuggingFace model with Modelium

This example shows how to:
1. Download a model from HuggingFace Hub
2. Deploy it using Modelium
3. Make predictions
"""

import requests
import time
from pathlib import Path


def deploy_from_hub(api_url: str, repo_id: str, model_name: str):
    """Deploy model directly from HuggingFace Hub."""
    
    print(f"Deploying {repo_id} from HuggingFace Hub...")
    
    response = requests.post(
        f"{api_url}/api/v1/models/from-hub",
        json={
            "repo_id": repo_id,
            "name": model_name,
            "deployment_config": {
                "target_environment": "kubernetes",
                "gpu_type": "nvidia-a100",
                "max_latency_ms": 100,
                "precision": "fp16",
                "batch_size": "dynamic",
            }
        }
    )
    
    response.raise_for_status()
    model_info = response.json()
    
    print(f"✓ Deployment started")
    print(f"  Model ID: {model_info['id']}")
    
    return model_info["id"]


def wait_for_deployment(api_url: str, model_id: str):
    """Wait for model deployment."""
    
    print("\nWaiting for deployment...")
    
    while True:
        response = requests.get(f"{api_url}/api/v1/models/{model_id}")
        model = response.json()
        
        print(f"  Status: {model['status']}")
        
        if model["status"] == "deployed":
            print("✓ Deployed!")
            return model
        elif model["status"] == "failed":
            print(f"✗ Failed: {model.get('error')}")
            return None
        
        time.sleep(10)


def test_inference(endpoint: str, text: str):
    """Test the deployed model."""
    
    print(f"\nTesting inference with text: '{text}'")
    
    response = requests.post(
        endpoint,
        json={"text": text}
    )
    
    result = response.json()
    print(f"✓ Result: {result}")
    
    return result


def main():
    """Main function."""
    
    API_URL = "http://localhost:8000"
    
    # Example 1: BERT base uncased
    print("=" * 60)
    print("Example 1: Deploying BERT-base-uncased")
    print("=" * 60)
    
    model_id = deploy_from_hub(
        API_URL,
        repo_id="bert-base-uncased",
        model_name="bert-base"
    )
    
    model = wait_for_deployment(API_URL, model_id)
    
    if model and model.get("endpoint"):
        test_inference(
            model["endpoint"],
            "Modelium is amazing for ML deployment!"
        )
    
    # Example 2: DistilBERT
    print("\n" + "=" * 60)
    print("Example 2: Deploying DistilBERT")
    print("=" * 60)
    
    model_id = deploy_from_hub(
        API_URL,
        repo_id="distilbert-base-uncased",
        model_name="distilbert"
    )
    
    model = wait_for_deployment(API_URL, model_id)
    
    # Example 3: GPT-2 Small
    print("\n" + "=" * 60)
    print("Example 3: Deploying GPT-2 Small")
    print("=" * 60)
    
    model_id = deploy_from_hub(
        API_URL,
        repo_id="gpt2",
        model_name="gpt2-small"
    )
    
    model = wait_for_deployment(API_URL, model_id)
    
    print("\n" + "=" * 60)
    print("All models deployed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

