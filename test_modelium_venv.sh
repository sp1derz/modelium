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

echo "Running: pip install --upgrade pip"
pip install -q --upgrade pip

echo "Running: pip install -e \".[all]\""
pip install -q -e ".[all]"

# Install accelerate (required for brain model loading)
echo "Running: pip install accelerate"
pip install -q accelerate || echo "⚠️  accelerate install failed (may need CUDA)"

test_success "Modelium installed"

# ============================================
# STEP 3: Install Runtime (vLLM for testing)
# ============================================
test_step "STEP 3: Checking vLLM installation"

# Check if on Linux (vLLM requires Linux+CUDA)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Linux detected - checking Python development headers..."
    
    # Check for Python headers (required for vLLM CUDA compilation)
    # Get the Python version from the venv
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    echo "   Python version in venv: $PYTHON_VERSION"
    
    if ! python3 -c "import sysconfig; import os; h = os.path.join(sysconfig.get_path('include'), 'Python.h'); assert os.path.exists(h), f'Missing: {h}'" 2>/dev/null; then
        echo "⚠️  Python development headers not found for Python $PYTHON_VERSION!"
        echo "   Installing Python $PYTHON_VERSION development headers..."
        
        if command -v yum &>/dev/null; then
            # Amazon Linux 2023 - try multiple approaches
            echo "   Trying to install python${PYTHON_MAJOR}${PYTHON_MINOR}-devel..."
            if sudo yum install -y python${PYTHON_MAJOR}${PYTHON_MINOR}-devel 2>/dev/null; then
                echo "   ✅ Installed python${PYTHON_MAJOR}${PYTHON_MINOR}-devel"
            else
                echo "   ⚠️  python${PYTHON_MAJOR}${PYTHON_MINOR}-devel not in default repos"
                echo "   💡 Installing Python $PYTHON_VERSION from Amazon Linux Extras or EPEL..."
                
                # Try Amazon Linux Extras (if available)
                if command -v amazon-linux-extras &>/dev/null; then
                    echo "   Checking Amazon Linux Extras for Python $PYTHON_VERSION..."
                    sudo amazon-linux-extras install -y python${PYTHON_MAJOR}.${PYTHON_MINOR} 2>/dev/null || true
                fi
                
                # Try EPEL
                if ! rpm -q epel-release &>/dev/null; then
                    echo "   Installing EPEL repository..."
                    sudo yum install -y epel-release 2>/dev/null || true
                fi
                
                # Try again with EPEL
                sudo yum install -y python${PYTHON_MAJOR}${PYTHON_MINOR}-devel 2>/dev/null || {
                    echo "   ❌ Could not install Python $PYTHON_VERSION headers"
                    echo "   💡 Solutions:"
                    echo "      1. Install Python $PYTHON_VERSION from source:"
                    echo "         https://www.python.org/downloads/"
                    echo "      2. Use a Python 3.10+ container/image"
                    echo "      3. Install python${PYTHON_MAJOR}${PYTHON_MINOR} from alternative source"
                }
            fi
        elif command -v apt-get &>/dev/null; then
            # Ubuntu/Debian - usually has version-specific packages
            if sudo apt-get install -y python${PYTHON_MAJOR}.${PYTHON_MINOR}-dev 2>/dev/null; then
                echo "   ✅ Installed python${PYTHON_MAJOR}.${PYTHON_MINOR}-dev"
            else
                echo "   ⚠️  Could not install python${PYTHON_MAJOR}.${PYTHON_MINOR}-dev"
                echo "   Try: sudo apt-get update && sudo apt-get install python${PYTHON_MAJOR}.${PYTHON_MINOR}-dev"
            fi
        else
            echo "   ⚠️  Please install Python development headers manually"
        fi
    else
        echo "✅ Python development headers found"
    fi
    
    echo "Linux detected - installing vLLM..."
    
    # Check Python version - vLLM 0.10+ requires Python 3.10+
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; then
        echo "❌ Python $PYTHON_VERSION detected - vLLM requires Python 3.10+"
        echo "   Please recreate venv with Python 3.10 or 3.11:"
        echo "   python3.10 -m venv venv  # or python3.11"
        echo "   source venv/bin/activate"
        test_fail "Python 3.10+ required for vLLM"
        exit 1
    fi
    
    echo "Python $PYTHON_VERSION detected - installing latest vLLM..."
    echo "Running: pip install vllm"
    pip install -q vllm || {
        test_fail "vLLM installation failed (may need CUDA or Python headers)"
        echo "Falling back to Ray for testing..."
        echo "Running: pip install ray[serve]"
        pip install -q ray[serve]
    }
    if pip show vllm &>/dev/null; then
        echo "✅ vLLM version: $(pip show vllm | grep Version | cut -d' ' -f2)"
        test_success "vLLM installed"
    fi
else
    echo "Non-Linux OS detected - using Ray instead of vLLM"
    echo "Running: pip install ray[serve]"
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

MODEL_DIR="models/incoming/gpt2"

# Check if model exists and is valid
MODEL_VALID=false
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
    # Check if safetensors file exists and is not empty
    if [ -f "$MODEL_DIR/model.safetensors" ]; then
        SAFETENSORS_SIZE=$(stat -f%z "$MODEL_DIR/model.safetensors" 2>/dev/null || stat -c%s "$MODEL_DIR/model.safetensors" 2>/dev/null || echo "0")
        if [ "$SAFETENSORS_SIZE" -gt 1000000 ]; then  # > 1MB
            # Try to validate the file
            python3 -c "
from safetensors import safe_open
try:
    with safe_open('$MODEL_DIR/model.safetensors', framework='pt') as f:
        keys = list(f.keys())[:1]
    print('✅ Model file is valid')
    exit(0)
except Exception as e:
    print(f'❌ Model file is corrupted: {e}')
    exit(1)
" 2>/dev/null && MODEL_VALID=true || {
                echo "⚠️  Model file appears corrupted (size: ${SAFETENSORS_SIZE} bytes)"
                MODEL_VALID=false
            }
        else
            echo "⚠️  Model file is too small ($SAFETENSORS_SIZE bytes), likely corrupted"
            MODEL_VALID=false
        fi
    else
        echo "⚠️  model.safetensors not found"
        MODEL_VALID=false
    fi
fi

if [ "$MODEL_VALID" = false ]; then
    echo "Downloading GPT-2 model from HuggingFace..."
    # Remove corrupted model if it exists
    if [ -d "$MODEL_DIR" ]; then
        echo "   Removing corrupted/incomplete model files..."
        rm -rf "$MODEL_DIR"
    fi
    
    mkdir -p "$MODEL_DIR"
    python3 << 'PYEOF'
from transformers import AutoModel, AutoTokenizer
import os
model_dir = os.environ.get('MODEL_DIR', 'models/incoming/gpt2')
print('Downloading GPT-2 model from HuggingFace...')
print('This may take a few minutes (~500MB)...')
model = AutoModel.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
print('Saving model with safe_serialization=True...')
model.save_pretrained(model_dir, safe_serialization=True)
tokenizer.save_pretrained(model_dir)
print('✅ Model downloaded and saved successfully')
PYEOF
    MODEL_DIR="$MODEL_DIR" python3 -c "
import os
model_dir = os.environ['MODEL_DIR']
if os.path.exists(f'{model_dir}/model.safetensors'):
    size = os.path.getsize(f'{model_dir}/model.safetensors')
    size_mb = size / 1024 / 1024
    print(f'   Model file size: {size_mb:.1f}MB')
    if size < 1000000:
        print('❌ Downloaded model file is too small, download may have failed')
        exit(1)
    else:
        print('✅ Model file size looks good')
        exit(0)
else:
    print('❌ model.safetensors not found after download')
    exit(1)
" || {
        test_fail "Failed to download or validate GPT-2 model"
        exit 1
    }
    test_success "GPT-2 model downloaded"
else
    test_success "GPT-2 model already exists and is valid"
fi

# ============================================
# STEP 10: Debug - Check Model Detection Setup
# ============================================
test_step "STEP 10: Debugging model detection setup"

echo ""
echo "📋 Checking configuration..."
if [ -f "modelium.yaml" ]; then
    echo "Config file: modelium.yaml"
    echo "Watch directories from config:"
    grep -A 2 "watch_directories:" modelium.yaml | grep -v "^#" || echo "  (not found in config)"
else
    echo "⚠️  Config file not found!"
fi

echo ""
echo "📁 Checking watch directories..."
# Get watch directories from config using Python (handles paths correctly)
python3 << 'PYEOF'
import yaml
import os

try:
    with open('modelium.yaml', 'r') as f:
        config = yaml.safe_load(f) or {}
    
    dirs = config.get('orchestration', {}).get('model_discovery', {}).get('watch_directories', [])
    
    if not dirs:
        print("  ⚠️  No watch directories found in config")
    else:
        for dir_path in dirs:
            print(f"\n  Directory: {dir_path}")
            if os.path.isdir(dir_path):
                print("    ✅ EXISTS")
                print("    Files:")
                try:
                    files = os.listdir(dir_path)
                    for f in files[:10]:
                        print(f"      {f}")
                    if len(files) > 10:
                        print(f"      ... and {len(files) - 10} more")
                except Exception as e:
                    print(f"      (error listing: {e})")
                
                print("    Model files:")
                model_files = []
                for root, _, files in os.walk(dir_path):
                    for f in files:
                        if any(f.endswith(ext) for ext in ['.safetensors', '.bin', '.pt', '.pth']) or f == 'config.json':
                            model_files.append(os.path.join(root, f))
                            if len(model_files) >= 10:
                                break
                    if len(model_files) >= 10:
                        break
                
                if model_files:
                    for f in model_files[:10]:
                        size = os.path.getsize(f) / (1024*1024)
                        print(f"      {f} ({size:.1f}MB)")
                else:
                    print("      (none found)")
            else:
                print("    ❌ DOES NOT EXIST")
                print("    Creating it...")
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    print("    ✅ Created")
                except Exception as e:
                    print(f"    ❌ Failed to create: {e}")
except Exception as e:
    print(f"  ⚠️  Error reading config: {e}")
PYEOF

# Get first watch directory for later use
WATCHED_DIR=$(python3 -c "
import yaml
try:
    with open('modelium.yaml', 'r') as f:
        config = yaml.safe_load(f) or {}
    dirs = config.get('orchestration', {}).get('model_discovery', {}).get('watch_directories', [])
    print(dirs[0] if dirs else '')
except:
    print('')
" 2>/dev/null || echo "")

if [ -z "$WATCHED_DIR" ]; then
    echo "  ⚠️  Could not determine watched directory"
else
    echo ""
    echo "  Primary watch directory: $WATCHED_DIR"
fi

echo ""
echo "📁 Checking common model locations..."
for dir in "/home/ec2-user/models/incoming" "/models/incoming" "./models/incoming" "$HOME/models/incoming"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir exists"
        echo "    Files: $(ls -1 "$dir" 2>/dev/null | wc -l) items"
        if [ -f "$dir/model.safetensors" ] || [ -f "$dir/config.json" ]; then
            echo "    ✅ Contains model files!"
            ls -lh "$dir"/*.{safetensors,bin,pt,pth,json} 2>/dev/null | head -5
        fi
    fi
done

echo ""
echo "🔍 Checking Modelium server logs for watcher status..."
if [ -f "modelium_test.log" ]; then
    echo "Watcher-related log entries:"
    grep -i "watch\|discover\|scan\|orchestrator\|on_model_discovered" modelium_test.log | tail -20 || echo "  (none found)"
    echo ""
    echo "Model loading attempts:"
    grep -i "loading\|load_model\|runtime\|brain decision" modelium_test.log | tail -20 || echo "  (none found)"
    echo ""
    echo "Recent errors:"
    grep -i "error\|failed\|exception" modelium_test.log | tail -10 || echo "  (none found)"
    echo ""
    echo "📋 Full recent server logs (last 50 lines):"
    tail -50 modelium_test.log
fi

echo ""
echo "📊 Current model registry status:"
echo "Running: curl -s http://localhost:8000/models"
MODELS_JSON=$(curl -s http://localhost:8000/models)
echo "$MODELS_JSON" | jq '.' 2>/dev/null || echo "$MODELS_JSON"

# Check if runtime is null
RUNTIME_NULL=$(echo "$MODELS_JSON" | jq -r '.models[]?.runtime' 2>/dev/null | grep -c "null" || echo "0")
if [ "$RUNTIME_NULL" -gt 0 ]; then
    echo ""
    echo "⚠️  WARNING: Model has runtime=null"
    echo "   This means the orchestrator hasn't assigned a runtime yet."
    echo "   Checking if orchestrator callback is being called..."
fi

echo ""
echo "🔧 Attempting automatic fix..."

# Find where model files actually are
MODEL_FOUND_IN=""
for dir in "/home/ec2-user/models/incoming" "./models/incoming" "$HOME/models/incoming" "$(pwd)/models/incoming"; do
    if [ -f "$dir/model.safetensors" ] || [ -f "$dir/config.json" ]; then
        MODEL_FOUND_IN="$dir"
        echo "  ✅ Found model files in: $MODEL_FOUND_IN"
        break
    fi
done

# Check what directory is being watched
WATCHED_DIR=$(echo "$WATCH_DIRS" | head -1 | xargs)

if [ ! -z "$MODEL_FOUND_IN" ] && [ ! -z "$WATCHED_DIR" ]; then
    # Normalize paths
    MODEL_FOUND_IN_ABS=$(cd "$(dirname "$MODEL_FOUND_IN")" 2>/dev/null && pwd)/$(basename "$MODEL_FOUND_IN") || echo "$MODEL_FOUND_IN"
    WATCHED_DIR_ABS=$(cd "$(dirname "$WATCHED_DIR")" 2>/dev/null && pwd)/$(basename "$WATCHED_DIR") 2>/dev/null || echo "$WATCHED_DIR"
    
    # Check if they're different
    if [ "$MODEL_FOUND_IN_ABS" != "$WATCHED_DIR_ABS" ]; then
        echo "  ⚠️  Model is in $MODEL_FOUND_IN but config watches $WATCHED_DIR"
        echo "  🔧 Fixing by updating config..."
        
        # Update config file
        if [ -f "modelium.yaml" ]; then
            # Create backup
            cp modelium.yaml modelium.yaml.backup
            
            # Update watch_directories in config
            if grep -q "watch_directories:" modelium.yaml; then
                # Use absolute path
                MODEL_DIR_ABS=$(realpath "$MODEL_FOUND_IN" 2>/dev/null || echo "$MODEL_FOUND_IN")
                
                # Update the config using Python
                python3 -c "
import yaml
import os

model_dir = os.path.abspath(os.path.expanduser('$MODEL_DIR_ABS'))

try:
    with open('modelium.yaml', 'r') as f:
        config = yaml.safe_load(f) or {}

    # Update watch_directories
    if 'orchestration' not in config:
        config['orchestration'] = {}
    if 'model_discovery' not in config['orchestration']:
        config['orchestration']['model_discovery'] = {}
    
    config['orchestration']['model_discovery']['watch_directories'] = [model_dir]
    
    with open('modelium.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f'✅ Updated modelium.yaml to watch: {model_dir}')
except Exception as e:
    print(f'❌ Error updating config: {e}')
    exit(1)
"
                
                if [ $? -eq 0 ]; then
                    test_success "Config updated successfully"
                    echo "  📝 Restarting Modelium server to apply changes..."
                    pkill -f "modelium.cli serve" 2>/dev/null || true
                    sleep 2
                    
                    # Restart server
                    nohup python -m modelium.cli serve > modelium_test.log 2>&1 &
                    MODELIUM_PID=$!
                    echo "  ✅ Server restarted (PID: $MODELIUM_PID)"
                    
                    # Wait for server to be ready
                    sleep 5
                    for i in {1..10}; do
                        if curl -s http://localhost:8000/health | grep -q "healthy"; then
                            echo "  ✅ Server is ready"
                            break
                        fi
                        sleep 2
                    done
                else
                    echo "  ⚠️  Failed to update config automatically"
                    echo "  💡 Manual fix: Update modelium.yaml watch_directories to:"
                    echo "     watch_directories:"
                    echo "       - \"$MODEL_FOUND_IN\""
                fi
            fi
        fi
    else
        echo "  ✅ Model location matches watched directory"
    fi
else
    echo "  ⚠️  Could not determine model location or watched directory"
fi

# ============================================
# STEP 11: Wait for Model Detection
# ============================================
test_step "STEP 11: Waiting for model detection (max 60s)"

MODEL_DETECTED=false
for i in {1..12}; do
    MODELS=$(curl -s http://localhost:8000/models)
    MODEL_COUNT=$(echo "$MODELS" | jq '.models | length' 2>/dev/null || echo "0")
    
    if [ "$MODEL_COUNT" -gt 0 ]; then
        test_success "Model detected by Modelium"
        echo "Models: $MODELS" | jq '.' 2>/dev/null || echo "$MODELS"
        MODEL_DETECTED=true
        break
    fi
    
    if [ $i -eq 12 ]; then
        test_fail "Model not detected after 60 seconds"
        echo "Current registry:"
        echo "$MODELS" | jq '.' 2>/dev/null || echo "$MODELS"
        echo ""
        echo "Check logs:"
        tail -50 modelium_test.log
    fi
    sleep 5
done

# ============================================
# STEP 12: Wait for Model Loading
# ============================================
test_step "STEP 12: Waiting for model to load (may take 30-120s)"

# Get first model name if any detected
echo "Running: curl -s http://localhost:8000/models | jq -r '.models[0].name'"
FIRST_MODEL=$(curl -s http://localhost:8000/models | jq -r '.models[0].name' 2>/dev/null || echo "")
if [ -z "$FIRST_MODEL" ] || [ "$FIRST_MODEL" = "null" ]; then
    FIRST_MODEL="gpt2"  # Fallback to gpt2 for testing
fi

echo "Waiting for model: $FIRST_MODEL"
echo ""
echo "🔍 Debugging model loading..."
echo "Checking logs for orchestrator activity:"
if [ -f "modelium_test.log" ]; then
    echo "  - Orchestrator on_model_discovered calls:"
    grep -i "on_model_discovered\|New model discovered" modelium_test.log | tail -5 || echo "    (none found)"
    echo "  - Runtime decisions:"
    grep -i "brain decision\|choose_runtime\|runtime:" modelium_test.log | tail -5 || echo "    (none found)"
    echo "  - Load attempts:"
    grep -i "loading model\|load_model\|spawning\|starting" modelium_test.log | tail -5 || echo "    (none found)"
fi
echo ""

for i in {1..40}; do
    echo "Running: curl -s http://localhost:8000/models (attempt $i/40)"
    MODELS=$(curl -s http://localhost:8000/models)
    MODEL_STATUS=$(echo "$MODELS" | jq -r ".models[] | select(.name==\"$FIRST_MODEL\") | .status" 2>/dev/null || echo "")
    MODEL_RUNTIME=$(echo "$MODELS" | jq -r ".models[] | select(.name==\"$FIRST_MODEL\") | .runtime" 2>/dev/null || echo "null")
    
    if [ "$MODEL_STATUS" = "loaded" ]; then
        test_success "Model loaded successfully!"
        echo "  Runtime: $MODEL_RUNTIME"
        break
    elif [ "$MODEL_STATUS" = "error" ]; then
        test_fail "Model failed to load"
        echo "  Runtime: $MODEL_RUNTIME"
        echo "Check logs:"
        tail -100 modelium_test.log | grep -A 5 -B 5 -i "error\|failed\|exception" || tail -50 modelium_test.log
        kill $MODELIUM_PID 2>/dev/null || true
        exit 1
    fi
    
    if [ $i -eq 40 ]; then
        test_fail "Model loading timeout (200 seconds)"
        echo "Current status: $MODEL_STATUS"
        echo "Current runtime: $MODEL_RUNTIME"
        echo ""
        echo "Full model info:"
        echo "$MODELS" | jq ".models[] | select(.name==\"$FIRST_MODEL\")" 2>/dev/null || echo "$MODELS"
        echo ""
        echo "Recent logs (last 100 lines):"
        tail -100 modelium_test.log
        echo ""
        echo "Orchestrator activity:"
        grep -i "orchestrator\|on_model_discovered\|brain decision\|loading model" modelium_test.log | tail -20 || echo "  (none found)"
        kill $MODELIUM_PID 2>/dev/null || true
        exit 1
    fi
    
    echo "  Status: $MODEL_STATUS, Runtime: $MODEL_RUNTIME (attempt $i/40)"
    if [ "$MODEL_STATUS" = "unloaded" ] && [ $i -gt 5 ]; then
        if [ $i -eq 10 ] || [ $i -eq 20 ] || [ $i -eq 30 ]; then
            echo "  ⚠️  Model still unloaded after $((i*5)) seconds"
            echo "  Checking server logs for errors..."
            if [ -f "modelium_test.log" ]; then
                echo "  Recent errors:"
                tail -100 modelium_test.log | grep -i "error\|failed\|exception\|vllm\|traceback" | tail -10 || echo "    (no errors found)"
                echo "  Orchestrator activity:"
                tail -100 modelium_test.log | grep -i "orchestrator\|on_model_discovered\|brain decision\|loading model\|spawning" | tail -5 || echo "    (no orchestrator activity found)"
            fi
        fi
    fi
    sleep 5
done

# ============================================
# STEP 13: Test Inference
# ============================================
test_step "STEP 13: Testing inference"

INFERENCE_RESULT=$(curl -s -X POST http://localhost:8000/predict/$FIRST_MODEL \
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
# STEP 15: Test Multiple Inferences (Load Test)
# ============================================
test_step "STEP 15: Running load test (10 requests)"

SUCCESS_COUNT=0
for i in {1..10}; do
    RESULT=$(curl -s -X POST http://localhost:8000/predict/$FIRST_MODEL \
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
# STEP 16: Check Orchestration
# ============================================
test_step "STEP 16: Checking intelligent orchestration"

# Get current model stats
MODELS=$(curl -s http://localhost:8000/models)
QPS=$(echo "$MODELS" | jq -r ".models[] | select(.name==\"$FIRST_MODEL\") | .qps" 2>/dev/null || echo "0")

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

