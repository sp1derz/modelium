# Modelium Architecture

## System Overview

Modelium is designed as a distributed, event-driven system that automates the entire lifecycle of ML model deployment, from ingestion to production serving.

## Core Components

### 1. Model Ingestion Service

**Responsibility**: Monitors input locations and triggers the analysis pipeline.

- **File Watching**: Monitors filesystem/S3 for new model artifacts
- **Event Generation**: Creates ingestion events for the pipeline
- **Artifact Storage**: Manages uploaded models in object storage
- **Metadata Tracking**: Records ingestion metadata in database

**Technologies**: FastAPI, Watchdog, Boto3, PostgreSQL

### 2. Model Analyzer (Core)

**Responsibility**: Analyzes model artifacts and generates descriptors.

**Sub-components**:
- **Framework Detector**: Identifies PyTorch, TensorFlow, ONNX, etc.
- **Operation Extractor**: Analyzes model graph and operations
- **Resource Estimator**: Calculates memory and compute requirements
- **Security Scanner**: Detects malicious code and license issues
- **Tokenizer Analyzer**: Extracts tokenizer configuration for NLP models

**Output**: ModelDescriptor JSON with complete model specification

### 3. Modelium LLM Service

**Responsibility**: Generates optimal conversion plans from descriptors.

**Architecture**:
- Fine-tuned Qwen-1.8B model trained on conversion examples
- Prompt engineering with system and user templates
- JSON output conforming to ConversionPlan schema
- FastAPI inference server with GPU acceleration

**Input**: ModelDescriptor + deployment requirements
**Output**: ConversionPlan with steps, configs, and tests

### 4. Plan Validator

**Responsibility**: Validates conversion plans before execution.

**Checks**:
- Schema conformance
- Dangerous command detection (whitelist/blacklist)
- Resource limit validation
- Dependency graph validation
- Script safety analysis

### 5. Sandboxed Executor

**Responsibility**: Executes conversion plans in isolated environments.

**Features**:
- Docker-in-Docker execution
- Network isolation (no external access)
- Resource quotas (CPU, memory, GPU)
- Timeout enforcement
- Artifact collection
- Comprehensive logging

**Isolation Layers**:
1. Container isolation (Docker)
2. Network policies (none mode)
3. Filesystem isolation (bind mounts)
4. Resource limits (cgroups)

### 6. Model Converters

**Responsibility**: Implement specific conversion logic.

**Components**:
- **PyTorchConverter**: PyTorch → TorchScript/ONNX
- **ONNXConverter**: ONNX optimization and validation
- **TensorRTConverter**: ONNX → TensorRT engines
- **QuantizationEngine**: FP16/INT8 quantization
- **LLMConverter**: HuggingFace → TRT-LLM

### 7. Deployment Engine

**Responsibility**: Deploys models to inference infrastructure.

**Components**:
- **TritonConfigGenerator**: Generates config.pbtxt
- **ModelRepositoryManager**: Manages Triton model repository
- **KServeManifestGenerator**: Creates InferenceService YAML
- **DeploymentOrchestrator**: Applies Kubernetes resources
- **HealthChecker**: Validates deployed models

### 8. Monitoring & Observability

**Responsibility**: Provides visibility into system operations.

**Stack**:
- **Prometheus**: Metrics collection from all services
- **Grafana**: Dashboards for visualization
- **OpenTelemetry**: Distributed tracing
- **Loki**: Log aggregation (optional)

**Key Metrics**:
- Model ingestion rate
- Conversion success/failure rates
- Conversion duration (p50, p95, p99)
- Deployment success rate
- Inference throughput and latency
- GPU utilization

## Data Flow

```
┌─────────────────┐
│  Model Artifact │
│   (File/S3)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Ingestion    │  ◄──── File watcher
│     Service     │
└────────┬────────┘
         │ Model path
         ▼
┌─────────────────┐
│  Model Analyzer │  ◄──── Framework detection
│   (Descriptor   │       Op extraction
│   Generator)    │       Security scan
└────────┬────────┘
         │ ModelDescriptor
         ▼
┌─────────────────┐
│   Meta-LLM      │  ◄──── Plan generation
│   (Conversion   │       Optimization selection
│    Planner)     │
└────────┬────────┘
         │ ConversionPlan
         ▼
┌─────────────────┐
│ Plan Validator  │  ◄──── Safety checks
│                 │       Resource validation
└────────┬────────┘
         │ Validated plan
         ▼
┌─────────────────┐
│   Sandboxed     │  ◄──── Docker execution
│   Executor      │       Artifact collection
└────────┬────────┘
         │ Converted model + configs
         ▼
┌─────────────────┐
│   Deployment    │  ◄──── Triton config
│    Engine       │       KServe manifest
└────────┬────────┘
         │ K8s resources
         ▼
┌─────────────────┐
│ Triton/KServe   │  ◄──── Model serving
│   Inference     │       Autoscaling
└─────────────────┘
```

## Security Architecture

### 1. Sandbox Isolation

**Layers**:
1. **Container Isolation**: Docker containers with minimal privileges
2. **Network Isolation**: No outbound network access
3. **Filesystem Isolation**: Read-only except workspace
4. **Resource Limits**: CPU, memory, and GPU quotas

### 2. Code Safety

**Mechanisms**:
- Command whitelist enforcement
- Dangerous pattern detection (rm -rf, curl | bash, etc.)
- Python script analysis (eval, exec, subprocess)
- Binary file scanning

### 3. Secrets Management

**Approach**:
- Kubernetes Secrets for credentials
- No secrets in container images
- Secret rotation support
- Audit logging of secret access

### 4. Network Security

**Policies**:
- Service-to-service mTLS (Istio)
- Network policies for ingress/egress
- No direct internet access from executors
- API authentication with JWT tokens

## Scalability

### Horizontal Scaling

**Components**:
- **Ingestion Service**: Stateless, scales with load
- **Executor Service**: Scales based on queue depth
- **Meta-LLM**: GPU-bound, limited by GPU availability
- **Deployment Service**: Stateless, scales easily

### Resource Management

**Strategies**:
- Queue-based workload distribution (Redis)
- Priority-based scheduling
- GPU sharing with time-slicing
- Spot instances for non-critical workloads

### Storage

**Approach**:
- S3-compatible object storage for models
- PostgreSQL with replication for metadata
- ReadWriteMany PVCs for shared workspaces
- Artifact lifecycle management (retention policies)

## High Availability

### Service Redundancy

- Multiple replicas for all services
- Active-active deployment pattern
- Health checks and automatic restart
- Rolling updates with zero downtime

### Data Durability

- PostgreSQL streaming replication
- S3 versioning and backup
- Disaster recovery procedures
- Point-in-time recovery capability

### Failure Handling

**Strategies**:
- Automatic retry with exponential backoff
- Circuit breakers for external services
- Graceful degradation
- Dead letter queues for failed jobs
- Fallback plans in conversion logic

## Performance Optimization

### Caching

- Model descriptor caching (Redis)
- Compiled TensorRT engines (S3)
- Frequently used base images (local registry)

### Parallelization

- Parallel model analysis
- Concurrent conversion execution
- Batch processing of small models

### Resource Pooling

- GPU sharing across conversions
- Container image reuse
- Persistent workspace volumes

## Monitoring & Alerting

### Key SLIs

1. **Availability**: 99.9% uptime
2. **Latency**: p95 < 30s for analysis, p95 < 10min for conversion
3. **Throughput**: 100+ models/hour
4. **Success Rate**: >95% successful conversions

### Alerting Rules

- Service down alerts (critical)
- High error rate (warning after 5min)
- Resource exhaustion (CPU/memory/GPU >90%)
- Long-running conversions (>1hr)
- Deployment failures

## Technology Stack Summary

| Component | Technologies |
|-----------|-------------|
| Application | Python 3.10, FastAPI, Pydantic |
| ML Frameworks | PyTorch, ONNX, TensorRT, TRT-LLM |
| Database | PostgreSQL, Redis |
| Storage | MinIO (S3-compatible) |
| Orchestration | Kubernetes, Helm, KServe |
| Inference | Triton Inference Server |
| Monitoring | Prometheus, Grafana, OpenTelemetry |
| CI/CD | GitHub Actions |
| Security | Docker isolation, Network policies |

## Future Enhancements

1. **Multi-cloud support**: AWS, GCP, Azure
2. **Model versioning**: A/B testing, canary deployments
3. **AutoML integration**: Hyperparameter optimization
4. **Federated learning**: Distributed model training
5. **Model marketplace**: Share and discover models
6. **Advanced optimizations**: Pruning, distillation, NAS

