#!/usr/bin/env python3
"""
Simple Modelium Example - Deploy Any Model in 3 Lines

This shows how easy it is to use Modelium:
1. Drop a model
2. Analyze it
3. Deploy it

Works on: AWS, GCP, Nebius, local GPU, even Mac CPU!
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modelium.core.analyzers import ModelAnalyzer
from modelium.modelium_llm.schemas import PlanGenerator
from modelium.runtimes.vllm_runtime import VLLMDeployment, VLLMConfig, create_vllm_config_from_descriptor
from modelium.runtimes.ray_serve import RayServeDeployment, RayServeConfig, create_ray_config_from_descriptor


def deploy_llm_example():
    """Example: Deploy an LLM with vLLM"""
    print("=" * 60)
    print("🚀 Example 1: Deploying LLM with vLLM")
    print("=" * 60)
    
    # Model from HuggingFace
    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model for demo
    organization_id = "my-org"
    
    print(f"\n📦 Model: {model_path}")
    print(f"🏢 Organization: {organization_id}")
    
    # Step 1: Analyze (optional, but shows what Modelium detects)
    print("\n🔍 Analyzing model...")
    # Note: For HuggingFace models, we create a descriptor manually
    descriptor = {
        "name": "tinyllama",
        "framework": "pytorch",
        "model_type": "llm",
        "resources": {"memory_bytes": 2_000_000_000},  # ~2GB
    }
    
    # Step 2: Generate deployment plan
    print("\n📋 Generating deployment plan...")
    plan = PlanGenerator.create_vllm_deployment_plan(
        model_id="tinyllama-1.1b",
        plan_id="plan-001",
        model_path=model_path,
        tensor_parallel_size=1,  # Use 1 GPU
    )
    
    print(f"   Target: {plan.target_format}")
    print(f"   Steps: {len(plan.steps)}")
    for i, step in enumerate(plan.steps, 1):
        print(f"   {i}. {step.name}: {step.description}")
    
    # Step 3: Generate deployment artifacts
    print("\n🛠️  Generating deployment files...")
    deployment = VLLMDeployment()
    config = VLLMConfig(
        model_path=model_path,
        model_name=f"{organization_id}-tinyllama",
        tensor_parallel_size=1,
        port=8000,
    )
    
    # Generate deployment script
    script = deployment.generate_deployment_script(config)
    script_file = Path("deploy_vllm.sh")
    script_file.write_text(script)
    print(f"   ✅ Created: {script_file}")
    
    # Generate docker-compose
    compose = deployment.generate_docker_compose(config)
    compose_file = Path("docker-compose-vllm.yml")
    compose_file.write_text(compose)
    print(f"   ✅ Created: {compose_file}")
    
    print("\n🎉 Ready to deploy!")
    print(f"\n   Run: bash {script_file}")
    print(f"   Or:  docker-compose -f {compose_file} up -d")
    print(f"\n   API: http://localhost:8000/v1/completions")
    print(f"   Docs: http://localhost:8000/docs")
    
    return script_file, compose_file


def deploy_pytorch_example():
    """Example: Deploy a PyTorch model with Ray Serve"""
    print("\n" + "=" * 60)
    print("🚀 Example 2: Deploying PyTorch Model with Ray Serve")
    print("=" * 60)
    
    import torch
    import torch.nn as nn
    
    organization_id = "my-org"
    
    # Create a simple model
    print("\n🔨 Creating test model...")
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(100, 50)
            self.fc2 = nn.Linear(50, 10)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    model = SimpleModel()
    model_path = Path("simple_model.pt")
    torch.save(model, model_path)
    print(f"   ✅ Created: {model_path}")
    
    # Step 1: Analyze
    print("\n🔍 Analyzing model...")
    analyzer = ModelAnalyzer()
    descriptor = analyzer.analyze(
        model_path,
        model_id="simple-model-001",
        model_name="simple-model"
    )
    print(f"   Framework: {descriptor.framework}")
    print(f"   Layers: {len(descriptor.layers)}")
    
    # Step 2: Generate deployment plan
    print("\n📋 Generating deployment plan...")
    plan = PlanGenerator.create_ray_serve_plan(
        model_id="simple-model-001",
        plan_id="plan-002",
        model_path=str(model_path),
        model_type="pytorch",
        convert_to_onnx=True,  # Convert to ONNX for better performance
    )
    
    print(f"   Target: {plan.target_format}")
    print(f"   Steps: {len(plan.steps)}")
    for i, step in enumerate(plan.steps, 1):
        print(f"   {i}. {step.name}: {step.description}")
    
    # Step 3: Generate deployment code
    print("\n🛠️  Generating deployment files...")
    deployment = RayServeDeployment()
    config = RayServeConfig(
        model_path=str(model_path),
        model_name=f"{organization_id}-simple-model",
        model_type="pytorch",
        num_replicas=1,
        num_gpus=0.0,  # CPU mode (works on Mac!)
        autoscaling=True,
        min_replicas=1,
        max_replicas=3,
    )
    
    # Generate deployment script
    deploy_code = deployment.generate_deployment_script(config)
    deploy_file = Path("deploy_ray.py")
    deploy_file.write_text(deploy_code)
    print(f"   ✅ Created: {deploy_file}")
    
    print("\n🎉 Ready to deploy!")
    print(f"\n   Run: python {deploy_file}")
    print(f"\n   API: http://localhost:8000/predict")
    print(f"\n   Test:")
    print(f"   curl -X POST http://localhost:8000/predict \\")
    print(f"     -H 'Content-Type: application/json' \\")
    print(f"     -d '{{\"input\": [[0.1, 0.2, ...]]}}' ")
    
    return deploy_file


def show_auto_selection():
    """Example: Let Modelium choose the best runtime"""
    print("\n" + "=" * 60)
    print("🧠 Example 3: Automatic Runtime Selection")
    print("=" * 60)
    
    import torch
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"\n💻 Available GPUs: {num_gpus}")
    
    # Test with different model types
    test_cases = [
        {"name": "llama-7b", "model_type": "llm", "memory_bytes": 7_000_000_000},
        {"name": "resnet50", "model_type": "vision", "memory_bytes": 100_000_000},
        {"name": "bert-base", "model_type": "text", "memory_bytes": 500_000_000},
    ]
    
    print("\n📊 Recommended runtimes:\n")
    for case in test_cases:
        descriptor = {
            "name": case["name"],
            "model_type": case["model_type"],
            "framework": "pytorch",
            "resources": {"memory_bytes": case["memory_bytes"]},
        }
        
        runtime = PlanGenerator.choose_best_runtime(descriptor, num_gpus)
        size_gb = case["memory_bytes"] / 1e9
        
        print(f"   {case['name']:15} ({size_gb:.1f}GB) → {runtime}")
    
    print("\n💡 Modelium automatically chooses:")
    print("   • vLLM for large language models (with GPU)")
    print("   • TensorRT for small models (max performance)")
    print("   • Ray Serve for general models (CPU or GPU)")


def main():
    """Run all examples"""
    print("\n🎯 Modelium - Simple Deployment Examples")
    print("=" * 60)
    print("\nThis shows how easy it is to deploy models with Modelium!")
    print("Works on AWS, GCP, Nebius, local GPU, even Mac CPU.\n")
    
    # Example 1: vLLM for LLMs
    try:
        vllm_files = deploy_llm_example()
    except Exception as e:
        print(f"\n⚠️  vLLM example skipped (install with: pip install vllm)")
        print(f"   Error: {e}")
    
    # Example 2: Ray Serve for PyTorch
    try:
        ray_file = deploy_pytorch_example()
    except Exception as e:
        print(f"\n⚠️  Ray example skipped")
        print(f"   Error: {e}")
    
    # Example 3: Auto-selection
    try:
        show_auto_selection()
    except Exception as e:
        print(f"\n⚠️  Auto-selection example skipped")
        print(f"   Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Examples Complete!")
    print("=" * 60)
    print("\n📚 What you learned:")
    print("   1. Deploy LLMs with vLLM (3 lines of code)")
    print("   2. Deploy PyTorch models with Ray Serve")
    print("   3. Let Modelium auto-choose the best runtime")
    print("\n🚀 Your models are ready for production!")
    print("\n💡 Multi-tenant? Just pass organizationId to track usage!")
    print("=" * 60)


if __name__ == "__main__":
    main()

