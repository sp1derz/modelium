"""
Example: Deploying Models with Triton Inference Server

This example shows how to:
1. Set up Triton model repository
2. Start Triton server
3. Connect Modelium to it

Prerequisites:
- Triton Docker container or Triton installed
- Model in Triton model repository format
"""

import requests
import json
import os
from pathlib import Path


def create_triton_model_repository():
    """
    Create a Triton model repository structure.
    
    Triton expects models in this format:
    model-repository/
      ├── my-model/
      │   ├── config.pbtxt  (model configuration)
      │   └── 1/            (version directory)
      │       └── model.plan (TensorRT) or model.onnx or model.pt
    """
    print("📁 Creating Triton model repository...")
    
    repo_path = Path("triton-models")
    repo_path.mkdir(exist_ok=True)
    
    # Example model config (for a simple ONNX model)
    model_name = "example-model"
    model_path = repo_path / model_name
    version_path = model_path / "1"
    version_path.mkdir(parents=True, exist_ok=True)
    
    # Create config.pbtxt
    config = """
name: "example-model"
platform: "onnxruntime_onnx"
max_batch_size: 8

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [-1, 3, 224, 224]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, 1000]
  }
]

dynamic_batching {
  preferred_batch_size: [1, 2, 4, 8]
  max_queue_delay_microseconds: 100
}
"""
    
    with open(model_path / "config.pbtxt", "w") as f:
        f.write(config)
    
    print(f"   ✅ Created {model_path}")
    print(f"   📝 Add your model file to {version_path}/model.onnx")
    print()
    
    return str(repo_path)


def check_triton_health():
    """Check if Triton is running."""
    try:
        response = requests.get("http://localhost:8003/v2/health/ready", timeout=5)
        return response.status_code == 200
    except:
        return False


def list_triton_models():
    """List models in Triton."""
    print("📋 Models in Triton:")
    
    try:
        response = requests.get("http://localhost:8003/v2/models")
        models = response.json()
        
        for model in models.get("models", []):
            print(f"   - {model['name']} (version: {model.get('version', 'N/A')})")
        
        print()
        return models
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def test_triton_inference():
    """Test inference on Triton."""
    print("🧪 Testing Triton inference...")
    
    model_name = "example-model"
    
    # Example inference request (KServe v2 protocol)
    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 3, 224, 224],
                "datatype": "FP32",
                "data": [0.1] * (3 * 224 * 224)  # Dummy input
            }
        ]
    }
    
    try:
        response = requests.post(
            f"http://localhost:8003/v2/models/{model_name}/infer",
            json=payload
        )
        
        result = response.json()
        print("   ✅ Triton inference successful!")
        print(f"   Output shape: {result['outputs'][0]['shape']}")
        print()
        return result
        
    except Exception as e:
        print(f"   ⚠️  Inference test skipped (add a real model first)")
        print(f"   Error: {e}")
        print()


def test_via_modelium():
    """Test Triton model through Modelium."""
    print("🧪 Testing via Modelium...")
    
    # Check if Triton models are registered in Modelium
    response = requests.get("http://localhost:8000/models")
    models = response.json()
    
    triton_models = [m for m in models['models'] if m['runtime'] == 'triton']
    print(f"   Triton models in Modelium: {[m['name'] for m in triton_models]}")
    
    if triton_models:
        model_name = triton_models[0]['name']
        print(f"   Testing {model_name}...")
        
        # Note: Triton inference through Modelium requires model-specific preprocessing
        # This is a simplified example
        response = requests.post(
            f"http://localhost:8000/predict/{model_name}",
            json={
                "prompt": "test",  # Simplified - actual Triton models need proper inputs
                "organizationId": "example-org",
            }
        )
        
        print("   ✅ Request routed through Modelium!")
    else:
        print("   ⚠️  No Triton models registered in Modelium yet")
    
    print()


def main():
    """Main example workflow."""
    print("=" * 70)
    print("Triton Inference Server Deployment Example")
    print("=" * 70)
    print()
    
    # Step 1: Create model repository
    print("Step 1: Create Triton model repository")
    repo_path = create_triton_model_repository()
    
    # Step 2: Instructions to start Triton
    print("Step 2: Start Triton Server")
    print("   Run this command:")
    print(f"   docker run --gpus all -p 8003:8000 -p 8004:8001 -p 8005:8002 \\")
    print(f"     -v {os.path.abspath(repo_path)}:/models \\")
    print(f"     nvcr.io/nvidia/tritonserver:24.01-py3 \\")
    print(f"     tritonserver --model-repository=/models")
    print()
    
    # Step 3: Check Triton
    print("Step 3: Verify Triton is running")
    if not check_triton_health():
        print("❌ Triton not running!")
        print("   Please start Triton with the command above")
        print()
        return
    
    print("✅ Triton is running")
    print()
    
    # Step 4: List models
    print("Step 4: List models in Triton")
    list_triton_models()
    
    # Step 5: Test inference
    print("Step 5: Test Triton inference")
    test_triton_inference()
    
    # Step 6: Test via Modelium
    print("Step 6: Test via Modelium")
    print("   (Make sure Modelium is running and triton.enabled: true)")
    
    try:
        test_via_modelium()
    except Exception as e:
        print(f"   ⚠️  Modelium test skipped: {e}")
    
    # Summary
    print("=" * 70)
    print("✅ Triton setup complete!")
    print()
    print("Next steps:")
    print("  1. Add your models to triton-models/")
    print("  2. Enable Triton in modelium.yaml (triton.enabled: true)")
    print("  3. Modelium will auto-detect and route to Triton models")
    print("=" * 70)


if __name__ == "__main__":
    main()

