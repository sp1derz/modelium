#!/usr/bin/env python3
"""
Show the Brain (Qwen) prompt that is used for orchestration decisions.

This script displays:
1. The system prompt (ORCHESTRATION_SYSTEM_PROMPT)
2. An example user prompt with Prometheus data
3. What metrics are sent to the brain
"""

from modelium.brain.prompts import ORCHESTRATION_SYSTEM_PROMPT, format_orchestration_prompt

print("=" * 80)
print("🧠 MODELIUM BRAIN PROMPT")
print("=" * 80)
print()
print("SYSTEM PROMPT (Loaded into Brain):")
print("-" * 80)
print(ORCHESTRATION_SYSTEM_PROMPT)
print()
print("=" * 80)
print()
print("EXAMPLE USER PROMPT (What gets sent with Prometheus data):")
print("-" * 80)

# Example current state (what Prometheus sends)
example_current_state = {
    "models_loaded": [
        {
            "name": "gpt2",
            "runtime": "ray",
            "gpu": 1,
            "qps": 2.5,  # From Prometheus: modelium_model_qps
            "idle_seconds": 15.3,  # From Prometheus: modelium_model_idle_seconds
            "loaded_at": 1700334000.0,
            "time_since_load_seconds": 120.5,
        },
        {
            "name": "gpt2-medium",
            "runtime": "ray",
            "gpu": 2,
            "qps": 0.0,  # Idle model
            "idle_seconds": 180.0,  # Idle for 3 minutes
            "loaded_at": 1700333880.0,
            "time_since_load_seconds": 240.5,
        }
    ],
    "gpu_memory_pressure": False,  # From PyTorch/nvidia-smi
    "total_gpus": 4,
}

# Example policies
example_policies = {
    "evict_after_idle_seconds": 300,  # 5 minutes
    "always_loaded": [],  # No models always loaded
    "evict_when_memory_above_percent": 85,
}

# Format the prompt
example_prompt = format_orchestration_prompt(
    current_state=example_current_state,
    policies=example_policies,
)

print(example_prompt)
print()
print("=" * 80)
print()
print("📊 PROMETHEUS METRICS SENT TO BRAIN:")
print("-" * 80)
print("Only relevant metrics are sent (not all Prometheus data):")
print()
print("For each model:")
print("  - qps: From modelium_model_qps gauge")
print("  - idle_seconds: From modelium_model_idle_seconds gauge")
print("  - gpu: Physical GPU ID")
print("  - time_since_load_seconds: Calculated from loaded_at timestamp")
print()
print("Global state:")
print("  - gpu_memory_pressure: From PyTorch/nvidia-smi (not Prometheus)")
print("  - total_gpus: From config")
print()
print("Policies:")
print("  - evict_after_idle_seconds: From config")
print("  - always_loaded: From config")
print("  - evict_when_memory_above_percent: From config")
print()
print("=" * 80)
print()
print("💡 The brain uses this data to decide:")
print("  - Keep models with high QPS")
print("  - Evict models with low QPS and long idle time")
print("  - Respect GPU memory pressure")
print("  - Follow policies (always_loaded, thresholds)")
print()

