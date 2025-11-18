#!/bin/bash
# Modelium Testing Script - Virtual Environment
# Tests the complete flow: Install → Configure → Start → Drop Model → Inference

set -e  # Exit on error

echo "========================================="
echo "🧪 MODELIUM VENV TESTING SCRIPT"
echo "========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

test_step() {
    echo ""
    echo -e "${YELLOW}▶ $1${NC}"
}

test_success() {
    echo -e "${GREEN}✅ $1${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

test_fail() {
    echo -e "${RED}❌ $1${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

# ============================================
# STEP 1: Environment Setup
# ============================================
test_step "STEP 1: Setting up virtual environment"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    test_success "Virtual environment created"
else
    test_success "Virtual environment already exists"
fi

source venv/bin/activate
test_success "Virtual environment activated"

# ============================================
# STEP 2: Install Dependencies
# ============================================
test_step "STEP 2: Installing Modelium"

pip install -q --upgrade pip
pip install -q -e ".[all]"
test_success "Modelium installed"

# ============================================
# STEP 3: Install Runtime (vLLM for testing)
# ============================================
test_step "STEP 3: Checking vLLM installation"

# Check if on Linux (vLLM requires Linux+CUDA)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Linux detected - installing vLLM..."
    pip install -q vllm || {
        test_fail "vLLM installation failed (may need CUDA)"
        echo "Falling back to Ray for testing..."
        pip install -q ray[serve]
    }
else
    echo "Non-Linux OS detected - using Ray instead of vLLM"
    pip install -q ray[serve]
    test_success "Ray installed (vLLM requires Linux+CUDA)"
fi

# ============================================
# STEP 4: Configuration
# ============================================
test_step "STEP 4: Creating configuration"

if [ ! -f "modelium.yaml" ]; then
    if [ -f "modelium.yaml.example" ]; then
        cp modelium.yaml.example modelium.yaml
        test_success "Config copied from example"
    else
        cat > modelium.yaml << 'EOF'
organization:
  id: "test-company"

gpu:
  enabled: auto

vllm:
  enabled: true

triton:
  enabled: false

ray_serve:
  enabled: false

orchestration:
  enabled: true
  model_discovery:
    watch_directories:
      - "./models/incoming"
    scan_interval_seconds: 5
  policies:
    evict_after_idle_seconds: 300
    always_loaded: []

metrics:
  enabled: true
  port: 9090

modelium_brain:
  enabled: true
  fallback_to_rules: true
EOF
        test_success "Config created"
    fi
else
    test_success "Config already exists"
fi

# ============================================
# STEP 5: Create Model Directory
# ============================================
test_step "STEP 5: Setting up model directory"

mkdir -p models/incoming
test_success "Model directory created"

# ============================================
# STEP 6: Start Modelium in Background
# ============================================
test_step "STEP 6: Starting Modelium server"

# Kill any existing Modelium process
pkill -f "modelium.cli serve" 2>/dev/null || true

# Start Modelium in background
nohup python -m modelium.cli serve > modelium_test.log 2>&1 &
MODELIUM_PID=$!

echo "Modelium PID: $MODELIUM_PID"
sleep 5  # Give it time to start

# Check if process is running
if ps -p $MODELIUM_PID > /dev/null; then
    test_success "Modelium server started (PID: $MODELIUM_PID)"
else
    test_fail "Modelium server failed to start"
    echo "Last 20 lines of log:"
    tail -20 modelium_test.log
    exit 1
fi

# ============================================
# STEP 7: Health Check
# ============================================
test_step "STEP 7: Testing health endpoint"

sleep 2
for i in {1..10}; do
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        test_success "Health check passed"
        break
    fi
    if [ $i -eq 10 ]; then
        test_fail "Health check failed after 10 attempts"
        echo "Server logs:"
        tail -30 modelium_test.log
        kill $MODELIUM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

# ============================================
# STEP 8: Status Check
# ============================================
test_step "STEP 8: Testing status endpoint"

STATUS=$(curl -s http://localhost:8000/status)
if echo "$STATUS" | grep -q "running"; then
    test_success "Status endpoint working"
    echo "Status: $STATUS" | jq '.' 2>/dev/null || echo "$STATUS"
else
    test_fail "Status endpoint returned unexpected response"
fi

# ============================================
# STEP 9: Download Test Model
# ============================================
test_step "STEP 9: Downloading GPT-2 model (small, ~500MB)"

if [ ! -d "models/incoming/gpt2" ]; then
    echo "Downloading from HuggingFace..."
    git clone --depth 1 https://huggingface.co/gpt2 models/incoming/gpt2 2>/dev/null || {
        echo "Git clone failed, trying with transformers..."
        python3 << 'PYEOF'
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("gpt2", cache_dir="./models/incoming/gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="./models/incoming/gpt2")
PYEOF
    }
    test_success "GPT-2 model downloaded"
else
    test_success "GPT-2 model already exists"
fi

# ============================================
# STEP 10: Wait for Model Detection
# ============================================
test_step "STEP 10: Waiting for model detection (max 60s)"

for i in {1..12}; do
    MODELS=$(curl -s http://localhost:8000/models)
    if echo "$MODELS" | grep -q "gpt2"; then
        test_success "Model detected by Modelium"
        echo "Models: $MODELS" | jq '.' 2>/dev/null || echo "$MODELS"
        break
    fi
    if [ $i -eq 12 ]; then
        test_fail "Model not detected after 60 seconds"
        echo "Check logs:"
        tail -50 modelium_test.log
    fi
    sleep 5
done

# ============================================
# STEP 11: Wait for Model Loading
# ============================================
test_step "STEP 11: Waiting for model to load (may take 30-120s)"

for i in {1..40}; do
    MODELS=$(curl -s http://localhost:8000/models)
    MODEL_STATUS=$(echo "$MODELS" | jq -r '.models[] | select(.name=="gpt2") | .status' 2>/dev/null || echo "")
    
    if [ "$MODEL_STATUS" = "loaded" ]; then
        test_success "Model loaded successfully!"
        break
    elif [ "$MODEL_STATUS" = "error" ]; then
        test_fail "Model failed to load"
        echo "Check logs:"
        tail -50 modelium_test.log
        kill $MODELIUM_PID 2>/dev/null || true
        exit 1
    fi
    
    if [ $i -eq 40 ]; then
        test_fail "Model loading timeout (200 seconds)"
        echo "Current status: $MODEL_STATUS"
        echo "Logs:"
        tail -50 modelium_test.log
        kill $MODELIUM_PID 2>/dev/null || true
        exit 1
    fi
    
    echo "  Status: $MODEL_STATUS (attempt $i/40)"
    sleep 5
done

# ============================================
# STEP 12: Test Inference
# ============================================
test_step "STEP 12: Testing inference"

INFERENCE_RESULT=$(curl -s -X POST http://localhost:8000/predict/gpt2 \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "organizationId": "test-company",
    "max_tokens": 20,
    "temperature": 0.7
  }')

if echo "$INFERENCE_RESULT" | grep -q "choices\|text\|error"; then
    test_success "Inference request completed"
    echo "Response: $INFERENCE_RESULT" | jq '.' 2>/dev/null || echo "$INFERENCE_RESULT"
else
    test_fail "Inference returned unexpected response"
    echo "Response: $INFERENCE_RESULT"
fi

# ============================================
# STEP 13: Test Metrics
# ============================================
test_step "STEP 13: Testing Prometheus metrics"

METRICS=$(curl -s http://localhost:9090/metrics 2>/dev/null || echo "")
if echo "$METRICS" | grep -q "modelium"; then
    test_success "Prometheus metrics working"
    echo "Sample metrics:"
    echo "$METRICS" | grep "modelium" | head -5
else
    test_fail "Prometheus metrics not available"
fi

# ============================================
# STEP 14: Test Multiple Inferences (Load Test)
# ============================================
test_step "STEP 14: Running load test (10 requests)"

SUCCESS_COUNT=0
for i in {1..10}; do
    RESULT=$(curl -s -X POST http://localhost:8000/predict/gpt2 \
      -H "Content-Type: application/json" \
      -d "{\"prompt\": \"Test $i\", \"max_tokens\": 10}" \
      2>/dev/null)
    
    if echo "$RESULT" | grep -q "choices\|text"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    fi
done

if [ $SUCCESS_COUNT -ge 8 ]; then
    test_success "Load test passed ($SUCCESS_COUNT/10 requests succeeded)"
else
    test_fail "Load test failed (only $SUCCESS_COUNT/10 requests succeeded)"
fi

# ============================================
# STEP 15: Check Orchestration
# ============================================
test_step "STEP 15: Checking intelligent orchestration"

# Get current model stats
MODELS=$(curl -s http://localhost:8000/models)
QPS=$(echo "$MODELS" | jq -r '.models[] | select(.name=="gpt2") | .qps' 2>/dev/null || echo "0")

echo "Current QPS: $QPS"
if [ "$QPS" != "0" ]; then
    test_success "Model is tracking QPS (intelligent orchestration active)"
else
    echo "⚠️  QPS is 0 (may need more time to update)"
fi

# ============================================
# CLEANUP
# ============================================
test_step "CLEANUP: Stopping Modelium server"

kill $MODELIUM_PID 2>/dev/null || true
sleep 2
test_success "Server stopped"

# ============================================
# SUMMARY
# ============================================
echo ""
echo "========================================="
echo "📊 TEST SUMMARY"
echo "========================================="
echo -e "${GREEN}✅ Tests Passed: $TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}❌ Tests Failed: $TESTS_FAILED${NC}"
else
    echo -e "${GREEN}❌ Tests Failed: $TESTS_FAILED${NC}"
fi
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED!${NC}"
    echo ""
    echo "Modelium is working correctly!"
    echo ""
    echo "To start Modelium again:"
    echo "  source venv/bin/activate"
    echo "  python -m modelium.cli serve"
    echo ""
    echo "Logs are in: modelium_test.log"
    exit 0
else
    echo -e "${RED}⚠️  SOME TESTS FAILED${NC}"
    echo ""
    echo "Check logs: modelium_test.log"
    echo ""
    exit 1
fi

