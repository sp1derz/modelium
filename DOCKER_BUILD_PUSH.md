# How to Build & Push Docker Image

Since GitHub Actions can't auto-build due to disk limits, here's how to build manually:

## Option 1: Build Locally & Push to GitHub Registry

### Prerequisites
- Docker installed locally
- GitHub Personal Access Token with `write:packages` scope
- ~20GB free disk space

### Steps

```bash
# 1. Clone repo (if you haven't)
git clone https://github.com/sp1derz/modelium.git
cd modelium

# 2. Build image (10-15 minutes)
docker build -t ghcr.io/sp1derz/modelium:latest .

# You'll see output like:
# [+] Building 847.3s (14/14) FINISHED
#  => [1/13] FROM docker.io/nvidia/cuda:12.1.0-runtime-ubuntu22.04
#  => [8/13] RUN poetry install --no-dev
# Final image size: ~8-10GB

# 3. Create GitHub Personal Access Token
# Go to: https://github.com/settings/tokens/new
# Scopes: write:packages, read:packages
# Save the token (you'll need it once)

# 4. Login to GitHub Container Registry
echo YOUR_GITHUB_TOKEN | docker login ghcr.io -u sp1derz --password-stdin

# Should see: Login Succeeded

# 5. Push image (5-10 minutes depending on upload speed)
docker push ghcr.io/sp1derz/modelium:latest

# You'll see:
# The push refers to repository [ghcr.io/sp1derz/modelium]
# latest: digest: sha256:abc123... size: 4523

# 6. Verify at: https://github.com/sp1derz?tab=packages
```

### Now Deploy

```bash
# Kubernetes will pull from ghcr.io
kubectl apply -k infra/k8s/

# Check image pull:
kubectl describe pod -n modelium modelium-server-xxx | grep -A 5 Events
```

---

## Option 2: Build on Cloud VM (No Registry Needed)

### For Single-Node Deployments

```bash
# 1. SSH to your VM
ssh ubuntu@your-ec2-ip

# 2. Install Docker + NVIDIA runtime (if not done)
# See DEPLOYMENT.md for full setup

# 3. Clone and build
git clone https://github.com/sp1derz/modelium.git
cd modelium
docker build -t modelium:latest .

# 4. Update deployment to use local image
sed -i 's|ghcr.io/sp1derz/modelium:latest|modelium:latest|' infra/k8s/deployment.yaml
sed -i 's|imagePullPolicy: Always|imagePullPolicy: IfNotPresent|' infra/k8s/deployment.yaml

# 5. Deploy
# Option A: Docker Compose (simplest)
docker-compose up -d

# Option B: Kubernetes (if you have K8s on VM)
kubectl apply -k infra/k8s/
```

---

## Option 3: Use Docker Hub

```bash
# 1. Build
docker build -t sp1derz/modelium:latest .

# 2. Login to Docker Hub
docker login
# Enter username: sp1derz
# Enter password: (your Docker Hub token)

# 3. Push
docker push sp1derz/modelium:latest

# 4. Update deployment
sed -i 's|ghcr.io/sp1derz/modelium|sp1derz/modelium|' infra/k8s/deployment.yaml

# 5. Deploy
kubectl apply -k infra/k8s/
```

---

## Option 4: Manual GitHub Actions Trigger

If you're feeling lucky (works ~50% of time):

1. Go to: https://github.com/sp1derz/modelium/actions/workflows/cd.yml
2. Click "Run workflow" button
3. Select branch: `main`
4. Click green "Run workflow"
5. Wait 10-15 minutes
6. If succeeds → image at ghcr.io/sp1derz/modelium:latest ✅
7. If fails with "no space left" → use Option 1-3 ❌

---

## Troubleshooting

### Error: `ImagePullBackOff` in Kubernetes

```bash
$ kubectl get pods -n modelium
NAME                          READY   STATUS             RESTARTS   AGE
modelium-server-xxx           0/1     ImagePullBackOff   0          2m

# Check what happened:
$ kubectl describe pod -n modelium modelium-server-xxx

Events:
  Warning  Failed     10s   kubelet  Failed to pull image "ghcr.io/sp1derz/modelium:latest": rpc error: code = NotFound
```

**Cause**: Image doesn't exist in registry

**Fix**: Build & push image (Option 1 or 3)

### Error: `ErrImageNeverPull` (for local images)

```bash
Events:
  Warning  Failed     10s   kubelet  Container image "modelium:latest" is not present with pull policy of Never
```

**Cause**: `imagePullPolicy: Never` but image not on node

**Fix**: Build image on the node:
```bash
# On the K8s node:
docker build -t modelium:latest /path/to/modelium/
```

### Error: During `docker push` - `unauthorized`

```bash
$ docker push ghcr.io/sp1derz/modelium:latest
unauthorized: authentication required
```

**Fix**: Login first:
```bash
echo YOUR_TOKEN | docker login ghcr.io -u sp1derz --password-stdin
```

---

## Which Option Should I Use?

### Use Option 1 (GitHub Registry) if:
- ✅ You have a multi-node Kubernetes cluster
- ✅ You want centralized image management
- ✅ You're okay with one-time manual build

### Use Option 2 (Build on VM) if:
- ✅ Single VM deployment
- ✅ Want to iterate quickly
- ✅ Don't need image registry

### Use Option 3 (Docker Hub) if:
- ✅ GitHub Container Registry has issues
- ✅ You already use Docker Hub
- ✅ Want public image availability

### Use Option 4 (GHA manual) if:
- ✅ You're feeling lucky
- ✅ Want to try automation
- ✅ Have 10-15 min to wait and see

---

## Recommended for Your Situation

Based on your setup (AWS g6e.12xlarge for testing):

**Use Option 2** - Build on the VM:

```bash
# Simplest workflow:
ssh ubuntu@your-ec2-ip
git clone https://github.com/sp1derz/modelium.git
cd modelium
docker-compose up -d

# That's it! No registry needed.
```

**Advantages**:
- Fast iteration (rebuild locally in 10 min)
- No registry complications
- Plenty of disk space on VM
- Simple debugging

**For production multi-node cluster**: Use Option 1 (one-time build + push)

---

## Summary

| Option | Time | Complexity | Best For |
|--------|------|------------|----------|
| 1. Build + Push to GHCR | 20-30 min | Medium | Multi-node K8s |
| 2. Build on VM | 10-15 min | Low | Single-node, iteration |
| 3. Build + Push to Docker Hub | 20-30 min | Medium | Multi-node K8s |
| 4. Manual GHA trigger | 10-15 min | Low | If lucky |

**For testing on g6e**: Use **Option 2**
**For production**: Use **Option 1**

