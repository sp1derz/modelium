# Modelium Deployment Guide

## Overview

Modelium can be deployed in multiple ways depending on your infrastructure:

1. **Local (Dev)** - Python CLI or Docker Compose
2. **Single VM** - Docker on EC2/GCE/Azure
3. **Kubernetes** - EKS/GKE/AKS with GPUs
4. **Helm** - Production-grade K8s deployment

---

## 1. Local Development

### Option A: Python CLI (Fastest)

```bash
# Create venv
python3.11 -m venv venv
source venv/bin/activate

# Install
pip install -e ".[all]"
pip install vllm  # Separate on Linux/CUDA

# Create config
python -m modelium.cli init

# Start server
python -m modelium.cli serve
```

### Option B: Docker Compose

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f modelium-server

# Stop
docker-compose down
```

**Access**: `http://localhost:8000`

---

## 2. Single VM Deployment (AWS/GCP/Azure)

### Prerequisites
- GPU instance (g6e.12xlarge, g5.12xlarge, etc.)
- Ubuntu 22.04 or Amazon Linux 2023
- NVIDIA drivers + CUDA 12.1+
- Docker with nvidia-container-toolkit

### Setup

```bash
# 1. Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 2. Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 3. Verify GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 4. Clone and configure
git clone https://github.com/sp1derz/modelium.git
cd modelium
cp modelium.yaml.example modelium.yaml
# Edit modelium.yaml

# 5. Start with Docker Compose
docker-compose up -d

# 6. Check health
curl http://localhost:8000/health
```

### Systemd Service (Production)

```bash
# Create systemd service
sudo tee /etc/systemd/system/modelium.service <<EOF
[Unit]
Description=Modelium Server
After=docker.service
Requires=docker.service

[Service]
Type=simple
WorkingDirectory=/opt/modelium
ExecStart=/usr/bin/docker-compose up
ExecStop=/usr/bin/docker-compose down
Restart=always
User=ubuntu

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl enable modelium
sudo systemctl start modelium
sudo systemctl status modelium
```

---

## 3. Kubernetes Deployment (Production)

### Prerequisites
- Kubernetes cluster with GPU nodes (EKS, GKE, AKS)
- `kubectl` configured
- NVIDIA GPU Operator installed
- Storage class for models (NFS, EBS, etc.)

### Quick Deploy with kubectl

```bash
# 1. Clone repo
git clone https://github.com/sp1derz/modelium.git
cd modelium

# 2. Update configs
# Edit infra/k8s/configmap.yaml - update organization, paths
# Edit infra/k8s/ingress.yaml - update domain
# Edit infra/k8s/pvc.yaml - update storageClass

# 3. Deploy
kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -f infra/k8s/rbac.yaml
kubectl apply -f infra/k8s/configmap.yaml
kubectl apply -f infra/k8s/pvc.yaml
kubectl apply -f infra/k8s/deployment.yaml
kubectl apply -f infra/k8s/service.yaml
kubectl apply -f infra/k8s/ingress.yaml

# 4. Check status
kubectl get pods -n modelium
kubectl logs -f deployment/modelium-server -n modelium

# 5. Port-forward for testing
kubectl port-forward -n modelium svc/modelium-server 8000:8000
```

### Deploy with Kustomize

```bash
# Edit kustomization
cd infra/k8s
vim kustomization.yaml  # Update image tag

# Apply
kubectl apply -k .

# Check
kubectl get all -n modelium
```

---

## 4. Helm Deployment (Recommended for Production)

### Install Helm Chart

```bash
# 1. Add chart repo (after publishing)
helm repo add modelium https://sp1derz.github.io/modelium-helm
helm repo update

# OR use local chart
cd infra/helm

# 2. Create values file
cat > my-values.yaml <<EOF
global:
  imageTag: "v0.2.0"

ingress:
  enabled: true
  hosts:
    - host: modelium.mydomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: modelium-tls
      hosts:
        - modelium.mydomain.com

resources:
  requests:
    nvidia.com/gpu: 4
  limits:
    nvidia.com/gpu: 4

persistence:
  models:
    storageClass: "fast-ssd"
    size: 500Gi

config:
  organization:
    id: "my-company"
  orchestration:
    enabled: true
    mode: "intelligent"
EOF

# 3. Install
helm install modelium ./modelium -f my-values.yaml --namespace modelium --create-namespace

# 4. Check status
helm status modelium -n modelium
kubectl get pods -n modelium

# 5. Upgrade
helm upgrade modelium ./modelium -f my-values.yaml -n modelium
```

### Helm Values Override Examples

**Staging (2 GPUs, less memory)**:
```yaml
resources:
  requests:
    memory: "16Gi"
    cpu: "4"
    nvidia.com/gpu: "2"
  limits:
    memory: "32Gi"
    cpu: "8"
    nvidia.com/gpu: "2"

config:
  organization:
    id: "staging"
  vllm:
    gpu_memory_utilization: 0.7
```

**Production (4 GPUs, high memory)**:
```yaml
resources:
  requests:
    memory: "32Gi"
    cpu: "8"
    nvidia.com/gpu: "4"
  limits:
    memory: "128Gi"
    cpu: "32"
    nvidia.com/gpu: "4"

config:
  organization:
    id: "production"
  orchestration:
    policies:
      evict_after_idle_seconds: 600  # 10 minutes
      always_loaded: ["critical-model-v1"]
```

---

## 5. Cloud-Specific Instructions

### AWS EKS

```bash
# 1. Create EKS cluster with GPU nodes
eksctl create cluster \
  --name modelium-cluster \
  --region us-west-2 \
  --nodegroup-name gpu-nodes \
  --node-type g5.12xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 4

# 2. Install NVIDIA GPU Operator
kubectl create namespace gpu-operator
helm install gpu-operator nvidia/gpu-operator -n gpu-operator

# 3. Deploy Modelium
helm install modelium ./infra/helm/modelium -n modelium --create-namespace

# 4. Use EBS for storage
# Update values.yaml:
persistence:
  models:
    storageClass: "gp3"  # AWS EBS gp3
```

### GCP GKE

```bash
# 1. Create GKE cluster
gcloud container clusters create modelium-cluster \
  --accelerator type=nvidia-l4,count=4 \
  --machine-type n1-standard-16 \
  --num-nodes 2 \
  --zone us-central1-a

# 2. Install GPU drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# 3. Deploy Modelium
helm install modelium ./infra/helm/modelium -n modelium --create-namespace
```

### Azure AKS

```bash
# 1. Create AKS cluster
az aks create \
  --resource-group modelium-rg \
  --name modelium-cluster \
  --node-count 2 \
  --vm-size Standard_NC6s_v3 \
  --node-vm-size Standard_NC6s_v3

# 2. Install GPU support
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml

# 3. Deploy Modelium
helm install modelium ./infra/helm/modelium -n modelium --create-namespace
```

---

## 6. Verification

### Check Health

```bash
# Direct
curl http://your-endpoint:8000/health

# Via kubectl
kubectl port-forward -n modelium svc/modelium-server 8000:8000
curl http://localhost:8000/health
```

### Check Status

```bash
curl http://your-endpoint:8000/status | jq .
```

Expected:
```json
{
  "status": "running",
  "organization": "my-company",
  "gpu_count": 4,
  "models_loaded": 0,
  "models_discovered": 0
}
```

### Test Model Deployment

```bash
# 1. Copy model to PVC
kubectl cp model.pt modelium/modelium-server-xxx:/models/incoming/

# 2. Watch logs
kubectl logs -f -n modelium deployment/modelium-server

# 3. Check models
curl http://your-endpoint:8000/models | jq .

# 4. Make prediction
curl -X POST http://your-endpoint:8000/predict/model \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "organizationId": "my-company"}'
```

---

## 7. Monitoring

### Prometheus Metrics

```bash
# Metrics endpoint
curl http://your-endpoint:9090/metrics
```

### Logs

```bash
# Docker
docker logs -f modelium-server

# Kubernetes
kubectl logs -f -n modelium deployment/modelium-server

# Helm
helm get notes modelium -n modelium
```

---

## 8. Troubleshooting

### Pod Pending

```bash
kubectl describe pod -n modelium modelium-server-xxx

# Check GPU availability
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.capacity.'nvidia\.com/gpu'
```

### Out of Memory

```bash
# Reduce GPU memory utilization in config
vllm:
  gpu_memory_utilization: 0.7  # Lower from 0.9

# Or reduce GPU request
resources:
  requests:
    nvidia.com/gpu: 2  # Lower from 4
```

### Model Loading Fails

```bash
# Check PVC is mounted
kubectl exec -it -n modelium modelium-server-xxx -- ls -la /models/incoming/

# Check logs
kubectl logs -n modelium modelium-server-xxx | grep ERROR
```

---

## 9. Scaling (Future)

Currently single-instance due to GPU orchestration complexity.

**Future multi-instance support**:
- Separate instances by model type (LLMs, vision)
- Load balancer with model routing
- Distributed orchestration

---

## 10. Uninstall

### Docker Compose
```bash
docker-compose down -v
```

### Kubectl
```bash
kubectl delete namespace modelium
```

### Helm
```bash
helm uninstall modelium -n modelium
kubectl delete namespace modelium
```

---

## Quick Reference

| Deployment | Time | Complexity | Use Case |
|------------|------|------------|----------|
| Python CLI | 5 min | Low | Local dev |
| Docker Compose | 10 min | Low | Testing |
| Single VM | 30 min | Medium | Small prod |
| Kubernetes | 1 hour | High | Enterprise |
| Helm | 30 min | Medium | Production |

---

For more help:
- [Architecture](docs/architecture.md)
- [Testing Guide](TESTING_TOMORROW.md)
- [Status](STATUS.md)

