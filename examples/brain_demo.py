"""
Demo: How the Unified Modelium Brain Works

This shows both tasks of the brain:
1. Conversion Planning (when new model is discovered)
2. Orchestration (every 10 seconds to manage GPU resources)
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modelium.brain import ModeliumBrain


def demo_conversion_planning():
    """
    Demo Task 1: Conversion Planning
    
    Scenario: User drops "qwen-7b.pt" in /models/incoming
    Brain analyzes it and decides how to deploy it.
    """
    print("=" * 80)
    print("DEMO: Task 1 - Conversion Planning")
    print("=" * 80)
    print()
    
    # Initialize brain (in production, this stays loaded)
    print("🧠 Initializing Modelium Brain...")
    brain = ModeliumBrain(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",  # Using public model for demo
        device="cuda:0" if __import__("torch").cuda.is_available() else "cpu",
        fallback_to_rules=True,  # Fall back to rules if LLM unavailable
    )
    print()
    
    # Simulate: User drops qwen-7b.pt
    print("📁 User drops: qwen-7b.pt → /models/incoming/")
    print("🔍 Modelium analyzes the model...")
    print()
    
    # Model descriptor (from analyzer)
    model_descriptor = {
        "name": "qwen-7b",
        "framework": "pytorch",
        "model_type": "causal_lm",
        "architecture": "Qwen2ForCausalLM",
        "parameters": 7_620_000_000,  # 7.6B params
        "resources": {
            "memory_bytes": 15_240_000_000,  # ~15GB (FP16)
        },
        "input_signature": {
            "input_ids": {"shape": [1, "seq_len"], "dtype": "int64"}
        },
        "output_signature": {
            "logits": {"shape": [1, "seq_len", 151936], "dtype": "float16"}
        }
    }
    
    # Current GPU state
    available_gpus = 4
    gpu_memory = [78, 52, 65, 26]  # Available GB per GPU
    
    print("💻 Current GPU State:")
    for i, mem in enumerate(gpu_memory):
        print(f"   GPU {i}: {mem}GB available")
    print()
    
    # Brain makes decision
    print("🧠 Brain analyzing...")
    plan = brain.generate_conversion_plan(
        model_descriptor=model_descriptor,
        available_gpus=available_gpus,
        gpu_memory=gpu_memory,
        target_environment="kubernetes",
    )
    
    print("✅ Deployment Plan Generated:")
    print(json.dumps(plan, indent=2))
    print()
    print(f"📊 Decision:")
    print(f"   Runtime: {plan['runtime']}")
    print(f"   Target GPU: {plan['target_gpu']}")
    print(f"   Reasoning: {plan['reasoning']}")
    print(f"   Confidence: {plan['confidence']:.2%}")
    print()


def demo_orchestration():
    """
    Demo Task 2: Orchestration
    
    Scenario: Brain checks every 10s and decides which models to keep/evict/load
    """
    print("=" * 80)
    print("DEMO: Task 2 - Orchestration Decision")
    print("=" * 80)
    print()
    
    # Initialize brain (same instance as above, already loaded)
    print("🧠 Using loaded Modelium Brain...")
    brain = ModeliumBrain(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device="cuda:0" if __import__("torch").cuda.is_available() else "cpu",
        fallback_to_rules=True,
    )
    print()
    
    # Current state (10 seconds ago)
    print("📊 Current State (10 models, 4 GPUs):")
    print()
    
    current_state = {
        "models_loaded": [
            {
                "name": "qwen-7b",
                "gpu": 1,
                "memory_gb": 15,
                "qps": 50,
                "latency_p99": 120,
                "idle_seconds": 0,
                "organization": "team-ml"
            },
            {
                "name": "bert-base",
                "gpu": 2,
                "memory_gb": 0.5,
                "qps": 0,
                "latency_p99": 0,
                "idle_seconds": 610,  # 10 minutes idle!
                "organization": "team-research"
            },
            {
                "name": "llama-13b",
                "gpu": 3,
                "memory_gb": 26,
                "qps": 20,
                "latency_p99": 180,
                "idle_seconds": 0,
                "organization": "team-product"
            },
            {
                "name": "resnet50",
                "gpu": 2,
                "memory_gb": 0.2,
                "qps": 5,
                "latency_p99": 30,
                "idle_seconds": 0,
                "organization": "team-product"
            },
        ],
        "models_unloaded": [
            {
                "name": "mistral-7b",
                "memory_gb": 14,
                "pending_requests": 3,
                "oldest_wait_seconds": 45,
                "organization": "team-ml"
            },
            {
                "name": "old-model-v1",
                "memory_gb": 8,
                "pending_requests": 0,
                "oldest_wait_seconds": 0,
                "organization": "team-research"
            }
        ],
        "gpu_memory": {
            "gpu_0": {"used": 2, "total": 80},  # Brain itself
            "gpu_1": {"used": 15, "total": 80},  # qwen-7b
            "gpu_2": {"used": 0.7, "total": 80},  # bert + resnet
            "gpu_3": {"used": 26, "total": 80},  # llama-13b
        }
    }
    
    # Print loaded models
    print("   Loaded Models:")
    for model in current_state["models_loaded"]:
        print(f"      • {model['name']} on GPU {model['gpu']}")
        print(f"        QPS: {model['qps']}, Idle: {model['idle_seconds']}s")
    print()
    
    # Print pending
    print("   Pending Models:")
    for model in current_state["models_unloaded"]:
        if model["pending_requests"] > 0:
            print(f"      • {model['name']}: {model['pending_requests']} requests waiting")
    print()
    
    # Print GPU usage
    print("   GPU Usage:")
    for gpu, mem in current_state["gpu_memory"].items():
        usage_pct = (mem["used"] / mem["total"]) * 100
        print(f"      {gpu}: {mem['used']:.1f}GB / {mem['total']}GB ({usage_pct:.0f}%)")
    print()
    
    # Policies
    policies = {
        "evict_after_idle_seconds": 300,  # 5 minutes
        "always_loaded": ["llama-13b"],  # SLA requirement
        "priority_by_qps": True,
        "priority_by_organization": True,
    }
    
    # Brain makes decision
    print("🧠 Brain analyzing current state...")
    decision = brain.make_orchestration_decision(
        current_state=current_state,
        policies=policies,
    )
    
    print()
    print("✅ Orchestration Decision:")
    print(json.dumps(decision, indent=2))
    print()
    
    # Print actions in human-readable format
    print("📋 Actions to Execute:")
    for i, action in enumerate(decision.get("actions", []), 1):
        action_type = action["action"]
        model_name = action["model"]
        reasoning = action.get("reasoning", "")
        
        if action_type == "evict":
            gpu = action.get("from_gpu", "?")
            print(f"   {i}. ❌ EVICT '{model_name}' from GPU {gpu}")
            print(f"      → {reasoning}")
        elif action_type == "load":
            gpu = action.get("to_gpu", "?")
            print(f"   {i}. ✅ LOAD '{model_name}' to GPU {gpu}")
            print(f"      → {reasoning}")
        elif action_type == "keep":
            gpu = action.get("on_gpu", "?")
            print(f"   {i}. ✓ KEEP '{model_name}' on GPU {gpu}")
            print(f"      → {reasoning}")
        
        print()
    
    print(f"🎯 Confidence: {decision.get('confidence', 0):.2%}")
    print()


def demo_full_workflow():
    """Show the complete workflow: discovery → conversion → orchestration."""
    print("=" * 80)
    print("FULL WORKFLOW: How Everything Works Together")
    print("=" * 80)
    print()
    
    print("Step 1: User drops 3 models")
    print("   📁 qwen-7b.pt → /models/incoming/")
    print("   📁 bert-base.pt → /models/incoming/")
    print("   📁 resnet50.pt → /models/incoming/")
    print()
    
    print("Step 2: Modelium discovers them (auto-scan every 30s)")
    print("   🔍 Analyzing qwen-7b.pt...")
    print("   🔍 Analyzing bert-base.pt...")
    print("   🔍 Analyzing resnet50.pt...")
    print()
    
    print("Step 3: Brain generates deployment plans (Task 1)")
    print("   🧠 qwen-7b → vLLM on GPU 1 (LLM detected)")
    print("   🧠 bert-base → Ray Serve on GPU 2 (small text model)")
    print("   🧠 resnet50 → TensorRT on GPU 2 (vision, max performance)")
    print()
    
    print("Step 4: Models deployed and ready")
    print("   ✅ http://localhost:8000/predict/qwen-7b")
    print("   ✅ http://localhost:8000/predict/bert-base")
    print("   ✅ http://localhost:8000/predict/resnet50")
    print()
    
    print("Step 5: Traffic comes in")
    print("   📈 qwen-7b: 50 QPS (high traffic!)")
    print("   📊 bert-base: 0 QPS (no one using it)")
    print("   📊 resnet50: 5 QPS")
    print()
    
    print("Step 6: 10 minutes later, bert-base still idle...")
    print("   ⏱️  bert-base: 610 seconds idle")
    print()
    
    print("Step 7: New request arrives for mistral-7b (not loaded)")
    print("   📥 Request queued: mistral-7b")
    print("   ⏳ Waiting for model to load...")
    print()
    
    print("Step 8: Brain makes orchestration decision (Task 2)")
    print("   🧠 Analyzing: 3 loaded, 1 pending, 1 idle...")
    print("   🧠 Decision:")
    print("      ❌ Evict bert-base (idle 10min, 0 QPS)")
    print("      ✅ Load mistral-7b to GPU 2 (pending request)")
    print("      ✓  Keep qwen-7b (high traffic)")
    print("      ✓  Keep resnet50 (active traffic)")
    print()
    
    print("Step 9: Actions executed")
    print("   ⚡ Unloading bert-base from GPU 2... (instant)")
    print("   ⚡ Loading mistral-7b to GPU 2... (30s with GDS)")
    print("   ✅ mistral-7b ready!")
    print()
    
    print("Step 10: Metrics updated")
    print("   📊 GPU Utilization: 68% → 74%")
    print("   📊 Avg Wait Time: 45s → 5s")
    print("   📊 Models Loaded: 3 (optimal for current traffic)")
    print()
    
    print("🎉 Result: Maximum GPU utilization, minimal latency!")
    print()


if __name__ == "__main__":
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "MODELIUM BRAIN DEMO" + " " * 39 + "║")
    print("║" + " " * 15 + "One LLM, Two Tasks, Maximum Efficiency" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Run demos
    demo_full_workflow()
    
    input("Press Enter to see detailed Task 1 demo (Conversion Planning)...")
    print()
    demo_conversion_planning()
    
    input("Press Enter to see detailed Task 2 demo (Orchestration)...")
    print()
    demo_orchestration()
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("The Modelium Brain (modelium/brain-v1) is a single fine-tuned Qwen-2.5-1.8B LLM that:")
    print()
    print("1. ✅ Analyzes new models and generates deployment plans")
    print("   • Chooses optimal runtime (vLLM, Ray, TensorRT)")
    print("   • Selects target GPU based on available memory")
    print("   • Configures settings (dtype, quantization, etc.)")
    print()
    print("2. ✅ Makes orchestration decisions every 10 seconds")
    print("   • Decides which models to keep loaded")
    print("   • Evicts idle models to free space")
    print("   • Loads pending models based on demand")
    print("   • Maximizes GPU utilization")
    print()
    print("3. ✅ Learns from patterns over time")
    print("   • Historical traffic patterns")
    print("   • Model usage by team/org")
    print("   • Time-of-day predictions")
    print()
    print("4. ✅ Falls back to rules if LLM unavailable")
    print("   • Simple heuristics as backup")
    print("   • System never breaks")
    print()
    print("Size: ~2GB VRAM, runs on any GPU")
    print("Model: Fine-tuned Qwen-2.5-1.8B")
    print("Location: HuggingFace (modelium/brain-v1)")
    print()
    print("💡 Users just need to:")
    print("   1. Configure modelium.yaml (one time)")
    print("   2. Drop models in /models/incoming")
    print("   3. Let the brain handle everything!")
    print()

