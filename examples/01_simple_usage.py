"""
Modelium - Simple Usage Example

This shows the SIMPLEST way to use Modelium:
1. Configure which runtimes you want
2. Drop models in a folder
3. That's it!
"""

import time
from pathlib import Path

# Step 1: Configure (modelium.yaml)
print("📝 Step 1: Configure")
print("""
# modelium.yaml
vllm:
  enabled: true   # ← Use vLLM for LLMs
triton:
  enabled: false
ray_serve:
  enabled: false

orchestration:
  model_discovery:
    watch_directories:
      - /models/incoming  # ← Watch this folder
  policies:
    evict_after_idle_seconds: 300  # ← Unload after 5 min idle
""")

# Step 2: Start Modelium
print("\n🚀 Step 2: Start Modelium Server")
print("""
python -m modelium.cli serve

# Output:
# 🧠 Starting Modelium Server...
# ✅ Server ready at http://0.0.0.0:8000
# 📊 Prometheus metrics at http://localhost:9090/metrics
""")

# Step 3: Drop a model
print("\n📦 Step 3: Drop a Model")
print("""
# Option A: Download from HuggingFace
git clone https://huggingface.co/gpt2 /models/incoming/gpt2

# Option B: Copy your own model
cp -r my-gpt2-model /models/incoming/

# Modelium automatically:
# 📋 Detects model
# 🔍 Analyzes config.json
# 🎯 Brain decides: GPT2 → vLLM
# 🚀 Spawns vLLM process
# ✅ Model loaded!
""")

# Step 4: Use it
print("\n💬 Step 4: Run Inference")
print("""
curl http://localhost:8000/predict/gpt2 \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "Hello, my name is",
    "max_tokens": 50,
    "temperature": 0.7
  }'

# Response:
# {
#   "choices": [
#     {
#       "text": "John and I am a software engineer..."
#     }
#   ]
# }
""")

# Step 5: Check metrics
print("\n📊 Step 5: Check Metrics")
print("""
curl http://localhost:9090/metrics

# Prometheus metrics:
# modelium_requests_total{model="gpt2",runtime="vllm"} 1
# modelium_latency_seconds{model="gpt2",p="50"} 0.123
# modelium_model_idle_seconds{model="gpt2"} 30.5
""")

# Step 6: Automatic unload
print("\n🔽 Step 6: Automatic Unload (After 5 min idle)")
print("""
# (No requests for 5 minutes)

# Modelium logs:
# 🔽 Unloading idle model: gpt2 (idle: 300s, QPS: 0.00)
# ✅ GPU freed!
""")

print("\n\n✨ THAT'S IT! Maximum GPU utilization with ZERO manual management.")
print("\nThe complexity is hidden. The user experience is simple:")
print("  1. Configure runtimes")
print("  2. Drop models")
print("  3. Done!")

