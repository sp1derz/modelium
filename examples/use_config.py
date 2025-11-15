#!/usr/bin/env python3
"""
Example: Using Modelium Configuration System

This shows how the config system works and how it helps you scale.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modelium.config import load_config, get_config
from modelium.core.analyzers import ModelAnalyzer
from modelium.modelium_llm.schemas import PlanGenerator


def example_1_basic_config():
    """Example 1: Load and use basic config"""
    print("=" * 60)
    print("Example 1: Basic Configuration")
    print("=" * 60)
    
    # Load config (looks for modelium.yaml in current directory)
    config = load_config()
    
    print(f"\n✅ Loaded config:")
    print(f"   Organization: {config.organization.id}")
    print(f"   Default Runtime: {config.runtime.default}")
    print(f"   GPU Enabled: {config.gpu.enabled}")
    print(f"   vLLM Port: {config.vllm.port}")
    print(f"   Ray Serve Port: {config.ray_serve.port}")


def example_2_runtime_selection():
    """Example 2: Automatic runtime selection based on model type"""
    print("\n" + "=" * 60)
    print("Example 2: Automatic Runtime Selection")
    print("=" * 60)
    
    config = get_config()
    organizationId = config.organization.id
    
    # Test different model types
    model_types = [
        ("llama-2-7b", "llm"),
        ("resnet50", "vision"),
        ("bert-base", "text"),
        ("whisper-large", "audio"),
    ]
    
    print(f"\n🧠 Automatic runtime selection for org: {organizationId}\n")
    
    for model_name, model_type in model_types:
        runtime = config.get_runtime_for_model(model_type, organizationId)
        port = config.get_port_for_runtime(runtime, model_type)
        
        print(f"   {model_name:20} (type: {model_type:8}) → {runtime:12} at port {port}")
    
    print("\n💡 Modelium automatically:")
    print("   - Chooses vLLM for LLMs (best throughput)")
    print("   - Chooses Ray Serve for vision/text (flexible)")
    print("   - Routes to correct port based on workload separation")


def example_3_workload_separation():
    """Example 3: Workload separation for scaling"""
    print("\n" + "=" * 60)
    print("Example 3: Workload Separation (Multi-Instance)")
    print("=" * 60)
    
    # Load enterprise config (if it exists)
    try:
        config = load_config("configs/enterprise-multi-workload.yaml")
    except:
        print("\n⚠️  Using default config (workload separation not enabled)")
        return
    
    if not config.workload_separation.enabled:
        print("\n⚠️  Workload separation not enabled in config")
        return
    
    print(f"\n✅ Workload separation enabled!")
    print(f"   Number of instances: {len(config.workload_separation.instances)}\n")
    
    # Show routing
    for instance_name, instance in config.workload_separation.instances.items():
        print(f"   {instance_name}:")
        print(f"      Description: {instance.description}")
        print(f"      Model Types: {', '.join(instance.model_types)}")
        print(f"      Runtime: {instance.runtime}")
        print(f"      GPUs: {instance.gpu_count}")
        print(f"      Ports: {8000 + instance.port_offset} (vLLM), {8001 + instance.port_offset} (Ray)")
        print()
    
    print("📊 How it works:")
    print("   1. Deploy each instance on a separate server/VM")
    print("   2. When user drops a model, Modelium routes to correct instance")
    print("   3. LLMs → llm_instance (server A)")
    print("   4. Vision → vision_instance (server B)")
    print("   5. Each scales independently!")


def example_4_organization_tracking():
    """Example 4: Multi-tenant organization tracking"""
    print("\n" + "=" * 60)
    print("Example 4: Multi-Tenant Organization Tracking")
    print("=" * 60)
    
    config = get_config()
    
    # Simulate different organizations deploying models
    organizations = [
        {"id": "startup-123", "tier": "basic"},
        {"id": "enterprise-456", "tier": "premium"},
        {"id": "mega-corp-789", "tier": "enterprise"},
    ]
    
    print(f"\n🏢 Usage tracking enabled: {config.usage_tracking.enabled}")
    print(f"   Tracking: {', '.join([k for k, v in config.usage_tracking.dict().items() if isinstance(v, bool) and v])}\n")
    
    for org in organizations:
        # Get rate limit for this org
        default_rpm = config.rate_limiting.per_organization.get("default_rpm", 1000)
        overrides = config.rate_limiting.per_organization.get("overrides", {})
        rpm = overrides.get(org["id"], default_rpm)
        
        print(f"   {org['id']:20} ({org['tier']:10}) → {rpm:,} req/min")
    
    print("\n💡 Multi-tenancy features:")
    print("   - Track usage per organization")
    print("   - Different rate limits per customer tier")
    print("   - Bill based on actual usage")
    print("   - All automatic with organizationId!")


def example_5_deployment_flow():
    """Example 5: Complete deployment flow using config"""
    print("\n" + "=" * 60)
    print("Example 5: Complete Deployment Flow with Config")
    print("=" * 60)
    
    config = get_config()
    organizationId = "demo-company"
    
    print(f"\n🎯 Scenario: User from '{organizationId}' drops a model\n")
    
    # Simulated model descriptor
    model_descriptor = {
        "name": "llama-2-7b",
        "model_type": "llm",
        "framework": "pytorch",
        "resources": {"memory_bytes": 14_000_000_000}  # ~14GB
    }
    
    print(f"1️⃣  Model detected: {model_descriptor['name']}")
    print(f"   Type: {model_descriptor['model_type']}")
    print(f"   Size: {model_descriptor['resources']['memory_bytes'] / 1e9:.1f}GB")
    
    # Step 1: Choose runtime from config
    runtime = config.get_runtime_for_model(model_descriptor['model_type'], organizationId)
    print(f"\n2️⃣  Config says: Use '{runtime}' runtime")
    
    # Step 2: Check if runtime is enabled
    if not config.is_runtime_enabled(runtime):
        print(f"   ❌ {runtime} is not enabled in config!")
        return
    
    print(f"   ✅ {runtime} is enabled")
    
    # Step 3: Get port for this runtime
    port = config.get_port_for_runtime(runtime, model_descriptor['model_type'])
    print(f"\n3️⃣  Deployment port: {port}")
    
    # Step 4: Get vLLM config settings
    if runtime == "vllm":
        vllm_cfg = config.vllm
        print(f"\n4️⃣  vLLM settings from config:")
        print(f"   Tensor Parallelism: {vllm_cfg.tensor_parallel_size or 'auto'}")
        print(f"   Quantization: {vllm_cfg.quantization or 'none'}")
        print(f"   GPU Memory: {vllm_cfg.gpu_memory_utilization * 100}%")
    
    # Step 5: Check rate limits
    rpm_limit = config.rate_limiting.requests_per_minute
    print(f"\n5️⃣  Rate limit for {organizationId}: {rpm_limit:,} req/min")
    
    # Step 6: Deploy!
    print(f"\n6️⃣  🚀 Deploying {model_descriptor['name']}...")
    print(f"   Organization: {organizationId}")
    print(f"   Runtime: {runtime}")
    print(f"   Endpoint: http://0.0.0.0:{port}/v1")
    print(f"   Usage tracking: {'✅ Enabled' if config.usage_tracking.enabled else '❌ Disabled'}")
    
    print("\n✅ Deployment complete! All settings came from modelium.yaml")


def main():
    """Run all examples"""
    print("\n🎯 Modelium Configuration System Examples")
    print("=" * 60)
    print("\nThese examples show how the config system makes scaling easy!")
    
    try:
        example_1_basic_config()
    except Exception as e:
        print(f"\n⚠️  Example 1 skipped: {e}")
    
    try:
        example_2_runtime_selection()
    except Exception as e:
        print(f"\n⚠️  Example 2 skipped: {e}")
    
    try:
        example_3_workload_separation()
    except Exception as e:
        print(f"\n⚠️  Example 3 skipped: {e}")
    
    try:
        example_4_organization_tracking()
    except Exception as e:
        print(f"\n⚠️  Example 4 skipped: {e}")
    
    try:
        example_5_deployment_flow()
    except Exception as e:
        print(f"\n⚠️  Example 5 skipped: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Examples Complete!")
    print("=" * 60)
    print("\n📝 Key Takeaways:")
    print("   1. One config file (modelium.yaml) controls everything")
    print("   2. Auto-selects best runtime per model type")
    print("   3. Supports workload separation (LLMs/vision on different servers)")
    print("   4. Multi-tenant with organizationId tracking")
    print("   5. Easy to scale from 1 server → 100s of servers")
    print("\n🚀 Configure once, scale forever!")
    print("=" * 60)


if __name__ == "__main__":
    main()

