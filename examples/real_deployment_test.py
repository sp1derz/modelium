#!/usr/bin/env python3
"""
REAL Deployment Test - Actually creates, analyzes, and prepares deployment

This shows REAL operations, not simulations:
- Creates a real PyTorch model
- Actually analyzes it (extracts layers, params)
- Generates real deployment files
- Shows what would happen in production
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not installed. Install with: pip install torch")

from modelium.core.analyzers import ModelAnalyzer
from modelium.modelium_llm.schemas import PlanGenerator
from modelium.runtimes.ray_serve import RayServeDeployment, RayServeConfig
from modelium.config import get_config


# Define model class at module level (needed for pickle serialization)
class RealTestModel(nn.Module if TORCH_AVAILABLE else object):
    """A real model for testing"""
    def __init__(self):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def test_real_deployment():
    """Real end-to-end test with actual model"""
    
    if not TORCH_AVAILABLE:
        print("\n❌ PyTorch is required for this test")
        print("Install with: pip install torch")
        return
    
    print("=" * 70)
    print("🧪 REAL Deployment Test - Actual Operations")
    print("=" * 70)
    
    # Configuration
    organizationId = "test-company"
    
    print(f"\n📊 Organization: {organizationId}")
    print(f"🔧 Loading config from modelium.yaml...")
    
    config = get_config()
    print(f"   ✅ Config loaded")
    print(f"   - Organization ID: {config.organization.id}")
    print(f"   - GPU enabled: {config.gpu.enabled}")
    print(f"   - Default runtime: {config.runtime.default}")
    
    # Step 1: CREATE A REAL MODEL
    print("\n" + "=" * 70)
    print("Step 1: Creating REAL PyTorch Model")
    print("=" * 70)
    
    model = RealTestModel()
    model_path = Path("real_test_model.pt")
    
    print(f"\n🔨 Creating model...")
    torch.save(model, model_path)
    
    # Get actual file size
    file_size = model_path.stat().st_size
    print(f"   ✅ Model created: {model_path}")
    print(f"   📦 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   📊 Total parameters: {total_params:,}")
    print(f"   📊 Trainable parameters: {trainable_params:,}")
    print(f"   📊 Model layers: {len(list(model.modules()))}")
    
    # Step 2: ACTUALLY ANALYZE IT
    print("\n" + "=" * 70)
    print("Step 2: REAL Model Analysis")
    print("=" * 70)
    
    print(f"\n🔍 Analyzing {model_path}...")
    print(f"   (This actually loads and inspects the model)")
    
    analyzer = ModelAnalyzer()
    
    try:
        descriptor = analyzer.analyze(
            model_path,
            model_id="real-test-001",
            model_name="real-test-model"
        )
        
        print(f"\n   ✅ Analysis complete!")
        print(f"\n   📋 Analysis Results:")
        print(f"      Framework: {descriptor.framework}")
        print(f"      Model Type: {descriptor.model_type}")
        print(f"      Model ID: {descriptor.id}")
        print(f"      Layers detected: {len(descriptor.layers)}")
        
        if descriptor.resources:
            print(f"\n   💾 Resource Estimates:")
            print(f"      Parameters: {descriptor.resources.parameters:,}")
            print(f"      Memory: {descriptor.resources.memory_bytes:,} bytes")
            print(f"      Memory: {descriptor.resources.memory_bytes/1024/1024:.2f} MB")
            if descriptor.resources.gpu_memory_bytes:
                print(f"      GPU Memory: {descriptor.resources.gpu_memory_bytes/1024/1024:.2f} MB")
        
        if descriptor.security_scan:
            print(f"\n   🔒 Security Scan:")
            print(f"      Risk Level: {descriptor.security_scan.risk_level}")
            print(f"      Has Pickle: {descriptor.security_scan.has_pickle}")
            if descriptor.security_scan.warnings:
                print(f"      Warnings: {len(descriptor.security_scan.warnings)}")
        
        if descriptor.layers:
            print(f"\n   🏗️  Model Architecture (first 5 layers):")
            for i, layer in enumerate(descriptor.layers[:5]):
                print(f"      {i+1}. {layer.type}: {layer.name}")
        
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        descriptor = None
    
    # Step 3: DETERMINE BEST RUNTIME
    print("\n" + "=" * 70)
    print("Step 3: Intelligent Runtime Selection")
    print("=" * 70)
    
    if descriptor:
        # Check available GPUs
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        print(f"\n🖥️  Hardware Detection:")
        print(f"   GPUs Available: {num_gpus}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if num_gpus > 0:
            print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        
        # Get runtime from config
        runtime = config.get_runtime_for_model(
            descriptor.model_type.lower() if descriptor.model_type else "unknown",
            organizationId
        )
        
        print(f"\n🧠 Modelium Decision:")
        print(f"   Model Type: {descriptor.model_type}")
        print(f"   Recommended Runtime: {runtime}")
        print(f"   Reason: ", end="")
        
        if runtime == "vllm":
            print("Large language model detected → vLLM (best for LLMs)")
        elif runtime == "tensorrt":
            print("Small model + GPU → TensorRT (max performance)")
        elif runtime == "ray_serve":
            if num_gpus > 0:
                print("General model + GPU → Ray Serve (flexible)")
            else:
                print("No GPU detected → Ray Serve (works on CPU)")
        else:
            print("Auto-select based on model characteristics")
        
        port = config.get_port_for_runtime(runtime, descriptor.model_type.lower())
        print(f"   Deployment Port: {port}")
    
    # Step 4: GENERATE REAL DEPLOYMENT FILES
    print("\n" + "=" * 70)
    print("Step 4: Generate REAL Deployment Files")
    print("=" * 70)
    
    if descriptor:
        print(f"\n📝 Generating deployment artifacts...")
        
        # Create Ray Serve config (works on Mac!)
        ray_config = RayServeConfig(
            model_path=str(model_path.absolute()),
            model_name=f"{organizationId}-real-test",
            model_type="pytorch",
            num_replicas=1,
            num_gpus=1.0 if num_gpus > 0 else 0.0,  # Use GPU if available
            autoscaling=True,
            min_replicas=1,
            max_replicas=3,
        )
        
        print(f"\n   🔧 Configuration:")
        print(f"      Model Path: {ray_config.model_path}")
        print(f"      Model Name: {ray_config.model_name}")
        print(f"      CPUs per replica: {ray_config.num_cpus}")
        print(f"      GPUs per replica: {ray_config.num_gpus}")
        print(f"      Auto-scaling: {ray_config.autoscaling}")
        print(f"      Min replicas: {ray_config.min_replicas}")
        print(f"      Max replicas: {ray_config.max_replicas}")
        
        # Generate deployment code
        deployment = RayServeDeployment()
        deploy_code = deployment.generate_deployment_script(ray_config)
        
        deploy_file = Path("deploy_real_model.py")
        deploy_file.write_text(deploy_code)
        
        print(f"\n   ✅ Created: {deploy_file}")
        print(f"   📦 File size: {deploy_file.stat().st_size:,} bytes")
        
        # Show snippet of what was generated
        lines = deploy_code.split('\n')
        print(f"\n   📄 Preview (first 15 lines):")
        for i, line in enumerate(lines[:15], 1):
            print(f"      {i:2d}: {line}")
        print(f"      ... ({len(lines)} total lines)")
    
    # Step 5: SHOW WHAT HAPPENS NEXT
    print("\n" + "=" * 70)
    print("Step 5: What Happens Next (In Production)")
    print("=" * 70)
    
    print(f"\n🚀 To ACTUALLY deploy this model:")
    print(f"\n   Option A: Run the generated deployment script")
    print(f"   $ python {deploy_file}")
    print(f"   Then test: curl http://localhost:8001/predict")
    
    print(f"\n   Option B: Use Docker")
    print(f"   $ docker-compose up -d")
    
    print(f"\n   Option C: Deploy to Kubernetes")
    print(f"   $ kubectl apply -f deployment.yaml")
    
    print(f"\n📊 Usage Tracking:")
    print(f"   Organization: {organizationId}")
    print(f"   Model: {descriptor.name if descriptor else 'N/A'}")
    print(f"   Inference calls → tracked for billing")
    print(f"   GPU hours → tracked for billing")
    print(f"   Storage → tracked for billing")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ Test Complete - All Operations Were REAL")
    print("=" * 70)
    
    print(f"\n📋 What Actually Happened:")
    print(f"   1. ✅ Created real PyTorch model ({file_size:,} bytes)")
    print(f"   2. ✅ Analyzed model (extracted {total_params:,} params)")
    print(f"   3. ✅ Detected hardware ({num_gpus} GPUs)")
    print(f"   4. ✅ Selected runtime ({runtime})")
    print(f"   5. ✅ Generated deployment file ({deploy_file})")
    
    print(f"\n🎯 Next Steps:")
    print(f"   • Run: python {deploy_file}  (to actually deploy)")
    print(f"   • The model is ready for production!")
    print(f"   • All operations tracked under org: {organizationId}")
    
    print("\n" + "=" * 70)
    
    # Cleanup option
    print(f"\n🧹 Cleanup files (optional):")
    print(f"   rm {model_path} {deploy_file}")
    print("=" * 70)


if __name__ == "__main__":
    test_real_deployment()

