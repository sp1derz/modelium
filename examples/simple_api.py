#!/usr/bin/env python3
"""
Modelium Simple API - Deploy ANY Model in 3 Lines

This is how users should actually use Modelium.
No boilerplate, just simple function calls.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Modelium
from modelium import get_config
from modelium.runtimes.vllm_runtime import VLLMDeployment, VLLMConfig
from modelium.runtimes.ray_serve import RayServeDeployment, RayServeConfig


def example_1_deploy_llm():
    """Example 1: Deploy an LLM with vLLM - 3 lines"""
    print("=" * 70)
    print("Example 1: Deploy LLM (3 lines of code)")
    print("=" * 70)
    
    print("\nCode you write:")
    print("""
    from modelium.runtimes.vllm_runtime import VLLMDeployment, VLLMConfig
    
    config = VLLMConfig(model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0", model_name="tinyllama")
    VLLMDeployment().deploy(config)
    # Done! API at http://localhost:8000/v1
    """)
    
    print("\nWhat Modelium does for you:")
    print("  ✅ Generates vLLM startup script")
    print("  ✅ Generates Docker Compose file")
    print("  ✅ Generates Kubernetes manifest")
    print("  ✅ Tracks under organizationId")
    print("  ✅ Sets up monitoring")
    
    print("\nYou never see the complexity!")


def example_2_deploy_pytorch():
    """Example 2: Deploy PyTorch model with Ray - 3 lines"""
    print("\n" + "=" * 70)
    print("Example 2: Deploy PyTorch Model (3 lines of code)")
    print("=" * 70)
    
    print("\nCode you write:")
    print("""
    from modelium.runtimes.ray_serve import RayServeDeployment, RayServeConfig
    
    config = RayServeConfig(model_path="model.pt", model_name="my-model", model_type="pytorch")
    RayServeDeployment().deploy(config)
    # Done! API at http://localhost:8001/predict
    """)
    
    print("\nWhat Modelium does for you:")
    print("  ✅ Generates 81-line Ray Serve deployment")
    print("  ✅ Sets up FastAPI endpoints")
    print("  ✅ Configures auto-scaling")
    print("  ✅ Handles GPU/CPU detection")
    print("  ✅ Tracks under organizationId")
    
    print("\nYou never write boilerplate!")


def example_3_auto_deploy():
    """Example 3: Automatic deployment - 1 line!"""
    print("\n" + "=" * 70)
    print("Example 3: Auto-Deploy with Config (1 line!)")
    print("=" * 70)
    
    print("\nCode you write:")
    print("""
    from modelium import deploy_model
    
    deploy_model("model.pt", organizationId="my-company")
    # That's it! Modelium handles everything!
    """)
    
    print("\nWhat Modelium does for you:")
    print("  ✅ Analyzes model (framework, size, type)")
    print("  ✅ Reads modelium.yaml config")
    print("  ✅ Chooses best runtime (vLLM/Ray/TensorRT)")
    print("  ✅ Generates all deployment files")
    print("  ✅ Deploys and starts serving")
    print("  ✅ Tracks everything under organizationId")
    
    print("\nComplete automation!")


def example_4_real_usage():
    """Example 4: REAL usage - what you'd actually do"""
    print("\n" + "=" * 70)
    print("Example 4: Real-World Usage")
    print("=" * 70)
    
    # Create a test model
    try:
        import torch
        import torch.nn as nn
        
        print("\n🔨 Creating test model...")
        model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))
        torch.save(model, "test_model.pt")
        print("   ✅ Created: test_model.pt")
        
        # Deploy with Modelium - 3 LINES!
        print("\n🚀 Deploying with Modelium (3 lines):")
        print("   config = RayServeConfig(model_path='test_model.pt', model_name='test', model_type='pytorch')")
        print("   deployment = RayServeDeployment()")
        print("   deployment.deploy(config)")
        
        config = RayServeConfig(
            model_path="test_model.pt",
            model_name="test-model",
            model_type="pytorch",
            num_gpus=0.0,  # CPU mode
        )
        
        deployment = RayServeDeployment()
        info = deployment.deploy(config)
        
        print(f"\n   ✅ Deployed!")
        print(f"   Endpoint: {info['endpoint']}")
        print(f"   Engine: {info['engine']}")
        
        # Show what was generated
        print(f"\n📄 Generated deployment code:")
        print(f"   File size: {len(info['deployment_code'])} characters")
        print(f"   Lines: {len(info['deployment_code'].split(chr(10)))}")
        print(f"\n   You never had to write this!")
        
        # Cleanup
        Path("test_model.pt").unlink()
        print("\n🧹 Cleaned up test file")
        
    except ImportError:
        print("\n⚠️  PyTorch not installed, skipping demo")


def main():
    """Show all examples"""
    print("\n" + "=" * 70)
    print("🎯 Modelium Simple API - How Users Actually Use It")
    print("=" * 70)
    
    print("\n💡 Key Idea:")
    print("   Users write 3 lines of code.")
    print("   Modelium generates 81+ lines of deployment code.")
    print("   Users never see the complexity!")
    
    example_1_deploy_llm()
    example_2_deploy_pytorch()
    example_3_auto_deploy()
    example_4_real_usage()
    
    print("\n" + "=" * 70)
    print("✅ Summary")
    print("=" * 70)
    
    print("\n📊 What Users Write:")
    print("   3 lines: import, config, deploy")
    
    print("\n📊 What Modelium Generates:")
    print("   81 lines: Ray Serve boilerplate")
    print("   OR 20 lines: vLLM startup script")
    print("   OR 50 lines: Kubernetes manifest")
    print("   OR 30 lines: Docker Compose")
    
    print("\n🎯 Result:")
    print("   Simple for users")
    print("   Production-ready under the hood")
    print("   organizationId tracking automatic")
    
    print("\n💡 The 81-line file you saw?")
    print("   That's what Modelium GENERATES for you.")
    print("   You write 3 lines, Modelium handles the rest!")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

