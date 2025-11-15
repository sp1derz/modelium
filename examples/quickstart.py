"""
Modelium Quickstart - The Simplest Possible Example

This shows how easy it is for users to:
1. Configure Modelium
2. Drop models
3. Let the brain handle everything
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modelium import ModeliumBrain

# ============================================================================
# THAT'S IT! This is all you need:
# ============================================================================

def main():
    print("🚀 Modelium Quickstart")
    print()
    
    # 1. Initialize the brain (ONE TIME, stays loaded)
    print("Step 1: Initialize the brain")
    print("   Code: brain = ModeliumBrain()")
    print()
    
    brain = ModeliumBrain(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",  # Downloads automatically from HF
        fallback_to_rules=True,  # Use simple rules if LLM fails
    )
    
    print()
    print("   ✅ Brain loaded and ready!")
    print()
    
    # 2. When a new model is dropped, brain analyzes it
    print("Step 2: User drops model → Brain analyzes it")
    print("   User: cp qwen-7b.pt /models/incoming/")
    print()
    
    # Simulate model descriptor (from analyzer)
    model_info = {
        "name": "qwen-7b",
        "framework": "pytorch",
        "model_type": "causal_lm",
        "parameters": 7_620_000_000,
        "resources": {"memory_bytes": 15_240_000_000},
    }
    
    plan = brain.generate_conversion_plan(
        model_descriptor=model_info,
        available_gpus=4,
        gpu_memory=[78, 52, 65, 26],
    )
    
    print(f"   🧠 Brain Decision:")
    print(f"      Runtime: {plan['runtime']}")
    print(f"      GPU: {plan['target_gpu']}")
    print(f"      Reasoning: {plan['reasoning']}")
    print()
    
    # 3. Every 10 seconds, brain manages GPU resources
    print("Step 3: Brain manages GPU resources (every 10s)")
    print()
    
    current_state = {
        "models_loaded": [
            {"name": "qwen-7b", "gpu": 1, "qps": 50, "idle_seconds": 0},
            {"name": "bert", "gpu": 2, "qps": 0, "idle_seconds": 610},
        ],
        "models_unloaded": [
            {"name": "mistral-7b", "pending_requests": 3},
        ],
        "gpu_memory": {
            "gpu_0": {"used": 2, "total": 80},
            "gpu_1": {"used": 15, "total": 80},
            "gpu_2": {"used": 0.5, "total": 80},
        }
    }
    
    decision = brain.make_orchestration_decision(
        current_state=current_state,
        policies={"evict_after_idle_seconds": 300},
    )
    
    print("   🧠 Brain Decision:")
    for action in decision["actions"]:
        if action["action"] == "evict":
            print(f"      ❌ Evict {action['model']}: {action['reasoning']}")
        elif action["action"] == "load":
            print(f"      ✅ Load {action['model']}: {action['reasoning']}")
    print()
    
    # Done!
    print("=" * 60)
    print("✅ That's it! The brain handles everything automatically:")
    print("   • Analyzes models")
    print("   • Chooses best runtime")
    print("   • Manages GPU resources")
    print("   • Maximizes utilization")
    print()
    print("📚 Next steps:")
    print("   1. Edit modelium.yaml (configure your setup)")
    print("   2. Run: modelium serve")
    print("   3. Drop models in /models/incoming/")
    print("   4. Make requests to http://localhost:8000/predict/<model>")
    print()

if __name__ == "__main__":
    main()
