# Curl Commands for Manual QPS Testing

Use these commands to manually send requests to models and test QPS tracking.

## Basic Request Format

```bash
curl -X POST http://localhost:8000/predict/{MODEL_NAME} \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Your prompt here",
    "max_tokens": 50,
    "organizationId": "test-company"
  }'
```

## Commands for Specific Models

### GPT-2 Medium

```bash
# Basic request
curl -X POST http://localhost:8000/predict/gpt2-medium \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_tokens": 50,
    "organizationId": "test-company"
  }'

# With more tokens
curl -X POST http://localhost:8000/predict/gpt2-medium \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "organizationId": "test-company"
  }'
```

### GPT-2 (Small)

```bash
# Basic request
curl -X POST http://localhost:8000/predict/gpt2 \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello world",
    "max_tokens": 50,
    "organizationId": "test-company"
  }'

# With more tokens
curl -X POST http://localhost:8000/predict/gpt2 \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The quick brown fox",
    "max_tokens": 100,
    "organizationId": "test-company"
  }'
```

## Quick Test Scripts

### Send Multiple Requests to gpt2-medium (to increase QPS)

```bash
# Send 10 requests quickly
for i in {1..10}; do
  curl -s -X POST http://localhost:8000/predict/gpt2-medium \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"Test request $i\", \"max_tokens\": 10, \"organizationId\": \"test-company\"}" \
    > /dev/null
  echo "Request $i sent"
  sleep 0.5
done
```

### Send Multiple Requests to gpt2 (to keep it active)

```bash
# Send 10 requests quickly
for i in {1..10}; do
  curl -s -X POST http://localhost:8000/predict/gpt2 \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"Active test $i\", \"max_tokens\": 10, \"organizationId\": \"test-company\"}" \
    > /dev/null
  echo "Request $i sent"
  sleep 0.5
done
```

## Check Model Status and QPS

```bash
# Check all models
curl -s http://localhost:8000/models | jq '.'

# Check specific model (gpt2-medium)
curl -s http://localhost:8000/models | jq '.models[] | select(.name=="gpt2-medium")'

# Check QPS for all models
curl -s http://localhost:8000/models | jq '.models[] | {name: .name, qps: .qps, idle_seconds: .idle_seconds, status: .status}'
```

## Continuous Monitoring

```bash
# Watch QPS in real-time (updates every 2 seconds)
watch -n 2 'curl -s http://localhost:8000/models | jq ".models[] | {name: .name, qps: .qps, idle: .idle_seconds, status: .status}"'
```

## Test Scenarios

### Scenario 1: Make gpt2-medium Active (Increase QPS)

```bash
# Send 5 requests quickly to increase QPS
for i in {1..5}; do
  curl -s -X POST http://localhost:8000/predict/gpt2-medium \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"QPS test $i\", \"max_tokens\": 20, \"organizationId\": \"test-company\"}" \
    > /dev/null
  sleep 0.3
done

# Check QPS immediately after
curl -s http://localhost:8000/models | jq '.models[] | select(.name=="gpt2-medium") | {qps: .qps, idle: .idle_seconds}'
```

### Scenario 2: Keep Both Models Active

```bash
# Send requests to both models alternately
for i in {1..10}; do
  # Request to gpt2
  curl -s -X POST http://localhost:8000/predict/gpt2 \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"Request $i to gpt2\", \"max_tokens\": 10, \"organizationId\": \"test-company\"}" \
    > /dev/null
  
  sleep 0.5
  
  # Request to gpt2-medium
  curl -s -X POST http://localhost:8000/predict/gpt2-medium \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"Request $i to gpt2-medium\", \"max_tokens\": 10, \"organizationId\": \"test-company\"}" \
    > /dev/null
  
  sleep 0.5
done
```

### Scenario 3: Single Request (Quick Test)

```bash
# Single request to gpt2-medium
curl -X POST http://localhost:8000/predict/gpt2-medium \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is artificial intelligence?",
    "max_tokens": 50,
    "organizationId": "test-company"
  }' | jq '.'
```

## Notes

- **QPS Calculation**: QPS is calculated over a 10-second sliding window
- **Idle Time**: Resets to 0 when a request is received
- **Grace Period**: Models are protected from eviction for 120 seconds after loading
- **Eviction Criteria**: Models are only evicted if QPS=0 AND idle>=180s AND grace period passed
- **organizationId**: Required field for multi-tenancy tracking

## Troubleshooting

If requests fail:
1. Check server is running: `curl http://localhost:8000/health`
2. Check model status: `curl http://localhost:8000/models`
3. Check logs: `tail -f modelium.log`

