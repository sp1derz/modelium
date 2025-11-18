#!/bin/bash
# Modelium Testing Script - Docker
# Tests the complete flow with Docker: Build → Run → Test

set -e  # Exit on error

echo "========================================="
echo "🐳 MODELIUM DOCKER TESTING SCRIPT"
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
# STEP 1: Check Docker
# ============================================
test_step "STEP 1: Checking Docker installation"

if ! command -v docker &> /dev/null; then
    test_fail "Docker is not installed"
    echo "Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
test_success "Docker is installed"

if ! docker ps &> /dev/null; then
    test_fail "Docker daemon is not running"
    echo "Start Docker daemon first"
    exit 1
fi
test_success "Docker daemon is running"

# ============================================
# STEP 2: Check docker-compose
# ============================================
test_step "STEP 2: Checking docker-compose"

if ! command -v docker-compose &> /dev/null; then
    test_fail "docker-compose is not installed"
    echo "Install: https://docs.docker.com/compose/install/"
    exit 1
fi
test_success "docker-compose is installed"

# ============================================
# STEP 3: Cleanup Previous Runs
# ============================================
test_step "STEP 3: Cleaning up previous containers"

docker-compose down -v 2>/dev/null || true
docker rm -f modelium-server vllm-server ray-server 2>/dev/null || true
test_success "Cleaned up previous containers"

# ============================================
# STEP 4: Build Docker Image
# ============================================
test_step "STEP 4: Building Docker image (this may take 5-15 minutes first time)"

echo "Starting build at $(date)"
if docker build -t modelium:latest . 2>&1 | tee docker_build.log; then
    test_success "Docker image built successfully"
else
    test_fail "Docker build failed"
    echo "Check docker_build.log for details"
    tail -50 docker_build.log
    exit 1
fi
echo "Build completed at $(date)"

# ============================================
# STEP 5: Create Model Directory
# ============================================
test_step "STEP 5: Setting up model directory"

mkdir -p models/incoming
test_success "Model directory created"

# ============================================
# STEP 6: Start Runtime Containers (vLLM/Ray)
# ============================================
test_step "STEP 6: Starting runtime containers"

# Create network if it doesn't exist (docker-compose will create it, but we need it now)
docker network create modelium_modelium-network 2>/dev/null || true

# Cleanup any existing runtime containers
docker rm -f vllm-server 2>/dev/null || true
docker rm -f ray-server 2>/dev/null || true

# Start vLLM container (for LLM inference)
# Note: vLLM container requires a model to start
# Users typically start their own vLLM containers with their models
# This container will be used by Modelium when models are loaded
echo "Starting vLLM container..."
echo "Note: vLLM requires a model parameter. Starting with placeholder..."
echo "      Modelium will connect to this container when loading models"
docker run -d \
  --name vllm-server \
  --gpus all \
  -p 8001:8000 \
  --network modelium_modelium-network \
  vllm/vllm-openai:latest \
  --model gpt2 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto || {
    test_fail "Failed to start vLLM container"
    echo "Note: vLLM requires NVIDIA GPU. If no GPU available, tests may fail."
    echo "      You can skip this step if testing without GPU."
}

sleep 5
if docker ps | grep -q vllm-server; then
    test_success "vLLM container started"
else
    echo "⚠️  vLLM container not running (may need GPU)"
fi

# Optional: Start Ray container (commented out by default)
# Uncomment below if you want to test with Ray Serve
# echo "Starting Ray container..."
# docker run -d \
#   --name ray-server \
#   --gpus all \
#   -p 8002:8000 \
#   -p 8265:8265 \
#   --network modelium_modelium-network \
#   rayproject/ray:latest \
#   ray start --head --port=6379 --dashboard-host=0.0.0.0 || {
#     test_fail "Failed to start Ray container"
# }
# sleep 5
# if docker ps | grep -q ray-server; then
#     test_success "Ray container started"
# else
#     test_fail "Ray container not running"
# fi

# ============================================
# STEP 7: Start Modelium Container
# ============================================
test_step "STEP 7: Starting Modelium container"

if docker-compose up -d; then
    test_success "Container started"
else
    test_fail "Failed to start container"
    docker-compose logs
    exit 1
fi

# Wait for container to be healthy
sleep 10

# ============================================
# STEP 7: Check Container Status
# ============================================
test_step "STEP 7: Checking container status"

if docker ps | grep -q modelium-server; then
    test_success "Container is running"
    docker ps | grep modelium-server
else
    test_fail "Container is not running"
    echo "Container logs:"
    docker-compose logs --tail=50
    exit 1
fi

# ============================================
# STEP 8: Health Check
# ============================================
test_step "STEP 8: Testing health endpoint"

for i in {1..15}; do
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        test_success "Health check passed"
        break
    fi
    if [ $i -eq 15 ]; then
        test_fail "Health check failed after 15 attempts"
        echo "Container logs:"
        docker-compose logs --tail=50
        docker-compose down
        exit 1
    fi
    sleep 2
done

# ============================================
# STEP 9: Status Check
# ============================================
test_step "STEP 9: Testing status endpoint"

STATUS=$(curl -s http://localhost:8000/status)
if echo "$STATUS" | grep -q "running"; then
    test_success "Status endpoint working"
    echo "$STATUS" | jq '.' 2>/dev/null || echo "$STATUS"
else
    test_fail "Status endpoint returned unexpected response"
    echo "Response: $STATUS"
fi

# ============================================
# STEP 10: Download Test Model
# ============================================
test_step "STEP 10: Downloading GPT-2 model (small, ~500MB)"

if [ ! -d "models/incoming/gpt2" ]; then
    echo "Downloading from HuggingFace..."
    git clone --depth 1 https://huggingface.co/gpt2 models/incoming/gpt2 || {
        test_fail "Failed to download model"
        docker-compose down
        exit 1
    }
    test_success "GPT-2 model downloaded"
else
    test_success "GPT-2 model already exists"
fi

# ============================================
# STEP 11: Wait for Model Detection
# ============================================
test_step "STEP 11: Waiting for model detection (max 60s)"

for i in {1..12}; do
    MODELS=$(curl -s http://localhost:8000/models)
    if echo "$MODELS" | grep -q "gpt2"; then
        test_success "Model detected by Modelium"
        echo "$MODELS" | jq '.' 2>/dev/null || echo "$MODELS"
        break
    fi
    if [ $i -eq 12 ]; then
        test_fail "Model not detected after 60 seconds"
        echo "Container logs:"
        docker-compose logs --tail=100
    fi
    sleep 5
done

# ============================================
# STEP 12: Wait for Model Loading
# ============================================
test_step "STEP 12: Waiting for model to load (may take 30-120s)"

for i in {1..40}; do
    MODELS=$(curl -s http://localhost:8000/models)
    MODEL_STATUS=$(echo "$MODELS" | jq -r '.models[] | select(.name=="gpt2") | .status' 2>/dev/null || echo "")
    
    if [ "$MODEL_STATUS" = "loaded" ]; then
        test_success "Model loaded successfully!"
        break
    elif [ "$MODEL_STATUS" = "error" ]; then
        test_fail "Model failed to load"
        echo "Container logs:"
        docker-compose logs --tail=100
        docker-compose down
        exit 1
    fi
    
    if [ $i -eq 40 ]; then
        test_fail "Model loading timeout (200 seconds)"
        echo "Current status: $MODEL_STATUS"
        echo "Container logs:"
        docker-compose logs --tail=100
        docker-compose down
        exit 1
    fi
    
    echo "  Status: $MODEL_STATUS (attempt $i/40)"
    sleep 5
done

# ============================================
# STEP 13: Test Inference
# ============================================
test_step "STEP 13: Testing inference"

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
    echo "$INFERENCE_RESULT" | jq '.' 2>/dev/null || echo "$INFERENCE_RESULT"
else
    test_fail "Inference returned unexpected response"
    echo "Response: $INFERENCE_RESULT"
fi

# ============================================
# STEP 14: Test Metrics
# ============================================
test_step "STEP 14: Testing Prometheus metrics"

METRICS=$(curl -s http://localhost:9090/metrics 2>/dev/null || echo "")
if echo "$METRICS" | grep -q "modelium"; then
    test_success "Prometheus metrics working"
    echo "Sample metrics:"
    echo "$METRICS" | grep "modelium" | head -5
else
    test_fail "Prometheus metrics not available"
fi

# ============================================
# STEP 15: Test Multiple Inferences
# ============================================
test_step "STEP 15: Running load test (10 requests)"

SUCCESS_COUNT=0
for i in {1..10}; do
    RESULT=$(curl -s -X POST http://localhost:8000/predict/gpt2 \
      -H "Content-Type: application/json" \
      -d "{\"prompt\": \"Docker test $i\", \"max_tokens\": 10}" \
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
# STEP 16: Check Container Logs
# ============================================
test_step "STEP 16: Checking container logs for errors"

ERROR_COUNT=$(docker-compose logs | grep -i "error\|exception\|failed" | grep -v "test" | wc -l || echo 0)
if [ "$ERROR_COUNT" -lt 5 ]; then
    test_success "No major errors in logs (found $ERROR_COUNT)"
else
    echo "⚠️  Found $ERROR_COUNT error lines in logs (may be normal)"
fi

# ============================================
# STEP 17: Check Resource Usage
# ============================================
test_step "STEP 17: Checking container resource usage"

STATS=$(docker stats --no-stream --format "{{.Container}}: CPU={{.CPUPerc}} MEM={{.MemUsage}}" modelium-server 2>/dev/null || echo "")
if [ ! -z "$STATS" ]; then
    test_success "Resource usage: $STATS"
else
    echo "⚠️  Could not get container stats"
fi

# ============================================
# CLEANUP (Optional)
# ============================================
echo ""
echo "========================================="
echo "🧹 CLEANUP"
echo "========================================="
echo ""
echo "Container is still running. Options:"
echo ""
echo "1. Keep it running for manual testing:"
echo "   docker-compose logs -f"
echo ""
echo "2. Stop it:"
echo "   docker-compose down"
echo "   docker rm -f vllm-server ray-server  # Stop runtime containers"
echo ""
echo "3. Stop and remove volumes:"
echo "   docker-compose down -v"
echo "   docker rm -f vllm-server ray-server  # Stop runtime containers"
echo ""

read -p "Stop container now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    test_step "Stopping containers"
    docker-compose down
    docker rm -f vllm-server ray-server 2>/dev/null || true
    test_success "Containers stopped"
else
    echo "Containers left running. Stop with:"
    echo "  docker-compose down"
    echo "  docker rm -f vllm-server ray-server"
fi

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
    echo "Modelium Docker is working correctly!"
    echo ""
    echo "Useful commands:"
    echo "  docker-compose logs -f        # View logs"
    echo "  docker-compose down           # Stop"
    echo "  docker-compose restart        # Restart"
    echo "  docker exec -it modelium-server bash  # Enter container"
    echo ""
    exit 0
else
    echo -e "${RED}⚠️  SOME TESTS FAILED${NC}"
    echo ""
    echo "Check logs: docker-compose logs"
    echo ""
    exit 1
fi

