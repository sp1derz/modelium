# CD Workflow Temporarily Disabled

## Why?

The CD (Continuous Deployment) workflow is disabled for automatic runs due to **GitHub Actions disk space limitations**.

### The Issue

- GitHub Actions runners: **14GB total disk space**
- Our Docker build requires: **~10GB** (CUDA 12.1 + PyTorch + vLLM)
- Build artifacts + cache: **~2-3GB**
- **Result**: "No space left on device" errors ❌

### What This Means

**CI (Continuous Integration)** ✅ Still works automatically:
- Code quality checks
- Import validation
- Security scanning
- Fast feedback (2-3 minutes)

**CD (Continuous Deployment)** ⚠️ Manual only:
- Docker builds
- Kubernetes deployment
- Must be triggered manually or built elsewhere

## How to Deploy

### Option 1: Manual GitHub Actions Trigger

```bash
# Go to GitHub → Actions → CD → Run workflow
# Click "Run workflow" button
# Select branch
```

This will:
- Build Docker image (if enough space available)
- Push to ghcr.io/sp1derz/modelium
- Deploy to staging/production

### Option 2: Build Locally (Recommended)

```bash
# Clone repo
git clone https://github.com/sp1derz/modelium.git
cd modelium

# Build image
docker build -t ghcr.io/sp1derz/modelium:latest .

# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Push
docker push ghcr.io/sp1derz/modelium:latest
```

### Option 3: Build on Cloud VM

Deploy directly on AWS/GCP/Azure:

```bash
# On your EC2/GCE/Azure VM
git clone https://github.com/sp1derz/modelium.git
cd modelium

# Option A: Docker Compose
docker-compose up -d

# Option B: Kubernetes
kubectl apply -k infra/k8s/

# Option C: Helm
helm install modelium ./infra/helm/modelium
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete cloud deployment guides.

## Solutions (Future)

### Short-term
1. ✅ Manual triggers (current solution)
2. Build on larger VMs (self-hosted runners)
3. Use external build service

### Long-term
1. Wait for GitHub to increase runner disk space
2. Multi-stage build optimization (saves ~500MB, not enough)
3. Self-hosted GitHub Actions runners with more disk
4. Use cloud build services (AWS CodeBuild, GCP Cloud Build)

## Why Not Make Image Smaller?

```
CUDA 12.1 runtime:     2GB   (needed for GPU)
PyTorch 2.7.1:         4GB   (needed for ML)
vLLM + dependencies:   2GB   (needed for LLM serving)
Modelium code:         100MB (our code)
---
Total:                 ~8-10GB (unavoidable)
```

**Can't reduce significantly** without:
- ❌ Removing CUDA (defeats purpose - no GPU support)
- ❌ Removing PyTorch (can't run models)
- ❌ Removing vLLM (can't serve LLMs efficiently)

CPU-only version would be ~2GB but would be 50-100x slower.

## Current Workaround

**For development/testing**:
- CI validates code automatically ✅
- Build Docker locally when needed
- Deploy to your own infrastructure

**For production**:
- Build once on cloud VM (plenty of disk space)
- Push to container registry
- Deploy via Kubernetes/Helm
- Updates: Rebuild on VM, not GitHub Actions

## When Will This Be Re-enabled?

We'll re-enable automatic CD when:
1. GitHub Actions increases disk space (>20GB), OR
2. We set up self-hosted runners with more disk, OR
3. We migrate to external build service

For now, **manual deployment works perfectly fine** for production use.

---

**Bottom line**: GitHub Actions disk limits are the bottleneck, not our code or Docker setup. 
Deploy using local builds or cloud VMs where disk space isn't an issue.

