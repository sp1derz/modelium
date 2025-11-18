"""
Example: Using Multiple Runtimes Simultaneously

This example demonstrates:
1. Running vLLM, Triton, and Ray Serve together
2. Modelium automatically routing to the right runtime
3. Monitoring and management across runtimes

Prerequisites:
- All runtimes running on different ports
- Models deployed to each runtime
"""

import requests
import json
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()


def check_runtime_health():
    """Check health of all runtimes."""
    runtimes = {
        "vLLM": {"url": "http://localhost:8001/health", "port": 8001},
        "Triton": {"url": "http://localhost:8003/v2/health/ready", "port": 8003},
        "Ray Serve": {"url": "http://localhost:8002/-/healthz", "port": 8002},
    }
    
    console.print("\n[bold]🏥 Runtime Health Check[/bold]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Runtime", style="dim", width=15)
    table.add_column("Status", width=10)
    table.add_column("Endpoint", width=30)
    
    available = []
    
    for name, info in runtimes.items():
        try:
            response = requests.get(info["url"], timeout=2)
            if response.status_code == 200:
                table.add_row(name, "✅ Healthy", info["url"])
                available.append(name)
            else:
                table.add_row(name, "⚠️  Unhealthy", info["url"])
        except:
            table.add_row(name, "❌ Down", info["url"])
    
    console.print(table)
    console.print()
    
    return available


def get_modelium_status():
    """Get Modelium server status."""
    try:
        response = requests.get("http://localhost:8000/status")
        status = response.json()
        
        console.print("[bold]🧠 Modelium Status[/bold]\n")
        console.print(f"  Status: {status['status']}")
        console.print(f"  Organization: {status['organization']}")
        console.print(f"  GPUs: {status['gpu_count']}")
        console.print(f"  Models Loaded: {status['models_loaded']}")
        console.print(f"  Models Discovered: {status['models_discovered']}")
        console.print()
        
        return status
    except:
        console.print("[red]❌ Modelium not running[/red]\n")
        return None


def list_models_by_runtime():
    """List models grouped by runtime."""
    try:
        response = requests.get("http://localhost:8000/models")
        data = response.json()
        models = data['models']
        
        console.print("[bold]📋 Models by Runtime[/bold]\n")
        
        # Group by runtime
        by_runtime = {}
        for model in models:
            runtime = model['runtime']
            if runtime not in by_runtime:
                by_runtime[runtime] = []
            by_runtime[runtime].append(model)
        
        for runtime, runtime_models in by_runtime.items():
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Model Name", style="dim")
            table.add_column("Status", width=12)
            table.add_column("GPU", width=6)
            table.add_column("QPS", width=8)
            
            for model in runtime_models:
                status_emoji = {
                    "loaded": "✅",
                    "loading": "⏳",
                    "error": "❌",
                    "discovered": "🔍"
                }.get(model['status'], "❓")
                
                table.add_row(
                    model['name'],
                    f"{status_emoji} {model['status']}",
                    str(model['gpu']) if model['gpu'] is not None else "-",
                    f"{model['qps']:.1f}"
                )
            
            console.print(f"[bold yellow]{runtime.upper()}[/bold yellow]")
            console.print(table)
            console.print()
        
        return models
        
    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]\n")
        return []


def test_inference_all_runtimes():
    """Test inference on models from each runtime."""
    console.print("[bold]🧪 Testing Inference Across Runtimes[/bold]\n")
    
    # Get models
    try:
        response = requests.get("http://localhost:8000/models")
        models = response.json()['models']
        
        # Test one model from each runtime
        tested_runtimes = set()
        
        for model in models:
            if model['status'] != 'loaded':
                continue
            
            runtime = model['runtime']
            if runtime in tested_runtimes:
                continue
            
            console.print(f"Testing {model['name']} ({runtime})...")
            
            # Inference request
            try:
                response = requests.post(
                    f"http://localhost:8000/predict/{model['name']}",
                    json={
                        "prompt": "Hello, world!",
                        "organizationId": "test-org",
                        "max_tokens": 20,
                        "temperature": 0.7,
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    console.print(f"  ✅ {runtime} inference successful")
                    tested_runtimes.add(runtime)
                else:
                    console.print(f"  ❌ {runtime} inference failed: {response.status_code}")
                    
            except Exception as e:
                console.print(f"  ❌ {runtime} inference error: {e}")
        
        console.print()
        console.print(f"Tested {len(tested_runtimes)} runtimes successfully")
        console.print()
        
    except Exception as e:
        console.print(f"[red]Error testing inference: {e}[/red]\n")


def demonstrate_auto_routing():
    """Demonstrate automatic routing to correct runtime."""
    console.print("[bold]🎯 Automatic Runtime Routing Demo[/bold]\n")
    
    console.print("Modelium automatically routes requests based on model type:")
    console.print("  - LLMs (GPT, Llama, etc.) → vLLM")
    console.print("  - Vision/General models → Ray Serve")
    console.print("  - Optimized models → Triton")
    console.print()
    
    console.print("Example: Drop models in /models/incoming/")
    console.print()
    console.print("  1. HuggingFace model with 'gpt' in name")
    console.print("     → Auto-detected as LLM")
    console.print("     → Routed to vLLM")
    console.print()
    console.print("  2. Vision model with 'vit' architecture")
    console.print("     → Auto-detected as Vision")
    console.print("     → Routed to Ray Serve")
    console.print()
    console.print("  3. Triton model in model repository")
    console.print("     → Auto-detected in Triton")
    console.print("     → Routed to Triton")
    console.print()
    
    console.print("All accessible via same API: /predict/{model-name}")
    console.print()


def show_startup_instructions():
    """Show how to start all runtimes."""
    console.print("[bold]🚀 Quick Start: All Runtimes[/bold]\n")
    
    console.print("[bold yellow]Terminal 1: vLLM (LLMs)[/bold yellow]")
    console.print("docker run --gpus all -p 8001:8000 vllm/vllm-openai:latest --model gpt2\n")
    
    console.print("[bold yellow]Terminal 2: Triton (All Models)[/bold yellow]")
    console.print("docker run --gpus all -p 8003:8000 -v ./triton-models:/models \\")
    console.print("  nvcr.io/nvidia/tritonserver:latest tritonserver --model-repository=/models\n")
    
    console.print("[bold yellow]Terminal 3: Ray Serve (Python Models)[/bold yellow]")
    console.print("docker run --gpus all -p 8002:8000 rayproject/ray:latest \\")
    console.print("  ray start --head\n")
    
    console.print("[bold yellow]Terminal 4: Modelium[/bold yellow]")
    console.print("python -m modelium.cli serve\n")
    
    console.print("[dim]Then drop models in models/incoming/ and watch them auto-deploy![/dim]\n")


def main():
    """Main demonstration."""
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]Multi-Runtime Deployment Example[/bold cyan]")
    console.print("=" * 70 + "\n")
    
    # Check what's running
    available = check_runtime_health()
    
    if not available:
        console.print("[yellow]No runtimes detected. Here's how to start them:[/yellow]\n")
        show_startup_instructions()
        return
    
    # Check Modelium
    status = get_modelium_status()
    if not status:
        console.print("[yellow]Start Modelium with: python -m modelium.cli serve[/yellow]\n")
        return
    
    # List models
    models = list_models_by_runtime()
    
    # Test inference
    if models:
        test_inference_all_runtimes()
    
    # Show routing demo
    demonstrate_auto_routing()
    
    # Summary
    console.print("=" * 70)
    console.print("[bold green]✅ Multi-Runtime Setup Working![/bold green]")
    console.print("=" * 70 + "\n")
    
    console.print("Try these commands:")
    console.print("  curl http://localhost:8000/models | jq")
    console.print("  curl http://localhost:8000/status | jq")
    console.print("  curl -X POST http://localhost:8000/predict/{model} -d '{...}'")
    console.print()


if __name__ == "__main__":
    main()

