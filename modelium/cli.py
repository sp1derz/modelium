"""
Modelium CLI tool.

Command-line interface for Modelium.
"""

import typer
import requests
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from pathlib import Path
from typing import Optional
import time
import sys

app = typer.Typer(help="Modelium - AI-Powered Model Serving Platform")
console = Console()


@app.command()
def serve(
    config: Path = typer.Option("modelium.yaml", "--config", help="Path to config file"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", help="Port to bind to"),
):
    """Start the Modelium server."""
    
    if not config.exists():
        console.print(f"[red]Error: Config file not found: {config}[/red]")
        console.print(f"Run 'modelium init' to create a default config")
        raise typer.Exit(1)
    
    console.print("[bold green]🧠 Starting Modelium Server...[/bold green]")
    console.print(f"Config: {config}")
    console.print(f"Host: {host}")
    console.print(f"Port: {port}")
    console.print()
    
    try:
        # Import here to avoid loading heavy deps at CLI startup
        from modelium.config import load_config
        from modelium.brain import ModeliumBrain
        
        # Load config
        console.print("📝 Loading configuration...")
        cfg = load_config(config)
        console.print(f"   Organization: {cfg.organization.id}")
        console.print(f"   GPUs: {cfg.gpu.count or 'auto-detect'}")
        console.print()
        
        # Initialize brain if enabled
        if cfg.modelium_brain.enabled:
            console.print("🧠 Loading Modelium Brain...")
            try:
                brain = ModeliumBrain(
                    model_name=cfg.modelium_brain.model_name,
                    device=cfg.modelium_brain.device,
                    dtype=cfg.modelium_brain.dtype,
                    fallback_to_rules=cfg.modelium_brain.fallback_to_rules,
                )
                console.print("   ✅ Brain loaded successfully")
            except Exception as e:
                if cfg.modelium_brain.fallback_to_rules:
                    console.print(f"   ⚠️  Brain load failed: {e}")
                    console.print("   ✅ Using rule-based mode (fallback)")
                else:
                    console.print(f"   ❌ Brain load failed: {e}")
                    raise typer.Exit(1)
        else:
            console.print("📊 Brain disabled, using rule-based mode")
        
        console.print()
        console.print("🚀 Server starting...")
        console.print(f"   API: http://{host}:{port}")
        console.print(f"   Status: http://{host}:{port}/status")
        console.print(f"   Metrics: http://{host}:9090/metrics")
        console.print()
        console.print("Press Ctrl+C to stop")
        console.print()
        
        # Initialize services
        console.print("🔧 Initializing services...")
        from modelium.services.model_registry import ModelRegistry
        from modelium.services.model_watcher import ModelWatcher
        from modelium.services.vllm_service import VLLMService
        from modelium.services.orchestrator import Orchestrator
        
        registry = ModelRegistry()
        vllm_service = VLLMService(host=host, port=cfg.vllm.port)
        
        # Create orchestrator
        orchestrator = Orchestrator(
            brain=brain if cfg.modelium_brain.enabled else None,
            vllm_service=vllm_service,
            config=cfg,
        )
        
        # Create model watcher
        watcher = ModelWatcher(
            watch_directories=cfg.orchestration.model_discovery.watch_directories,
            scan_interval=cfg.orchestration.model_discovery.scan_interval_seconds,
            supported_formats=cfg.orchestration.model_discovery.supported_formats,
            on_model_discovered=orchestrator.on_model_discovered,
        )
        
        # Start FastAPI server
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        
        app = FastAPI(title="Modelium API")
        
        class InferenceRequest(BaseModel):
            prompt: str
            organizationId: str
            max_tokens: int = 100
            temperature: float = 0.7
        
        @app.get("/")
        async def root():
            return {"message": "Modelium API", "version": "0.1.0"}
        
        @app.get("/status")
        async def status():
            import torch
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            stats = registry.get_stats()
            return {
                "status": "running",
                "organization": cfg.organization.id,
                "gpu_count": gpu_count,
                "gpu_enabled": cfg.gpu.enabled,
                "brain_enabled": cfg.modelium_brain.enabled,
                "orchestration_enabled": cfg.orchestration.enabled,
                "models_loaded": stats["loaded"],
                "models_discovered": stats["total_models"],
                "models_loading": stats["loading"],
            }
        
        @app.get("/models")
        async def list_models():
            """List all models."""
            models = registry.list_models()
            return {
                "models": [
                    {
                        "name": m.name,
                        "status": m.status.value,
                        "runtime": m.runtime,
                        "gpu": m.target_gpu,
                        "qps": m.qps,
                        "idle_seconds": m.idle_seconds,
                    }
                    for m in models
                ]
            }
        
        @app.post("/predict/{model_name}")
        async def predict(model_name: str, request: InferenceRequest):
            """Run inference on a model."""
            model = registry.get_model(model_name)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            
            if model.status != "loaded":
                raise HTTPException(
                    status_code=503, 
                    detail=f"Model not loaded (status: {model.status})"
                )
            
            # Record request
            registry.record_request(model_name)
            
            # Run inference
            result = vllm_service.inference(
                model_name=model_name,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            
            return result
        
        @app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        # Start background services
        console.print("🚀 Starting background services...")
        watcher.start()
        if cfg.orchestration.enabled:
            orchestrator.start()
        
        # Run server
        console.print("[green]✅ Server ready![/green]")
        console.print()
        uvicorn.run(app, host=host, port=port, log_level="info")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def check(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
):
    """Check system requirements and dependencies."""
    
    console.print("[bold]🔍 Modelium System Check[/bold]\n")
    
    all_ok = True
    
    # Python version
    import sys
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        console.print(f"✅ Python {py_version}")
    else:
        console.print(f"❌ Python {py_version} (require 3.10+)")
        all_ok = False
    
    # Core dependencies
    deps = {
        "torch": "PyTorch",
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "pydantic": "Pydantic",
        "transformers": "Transformers (for Brain)",
    }
    
    console.print("\n[bold]Core Dependencies:[/bold]")
    for module, name in deps.items():
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            console.print(f"✅ {name}: {version}")
        except ImportError:
            console.print(f"❌ {name}: Not installed")
            all_ok = False
    
    # Optional dependencies
    console.print("\n[bold]Optional Dependencies:[/bold]")
    optional = {
        "vllm": "vLLM (for LLM serving)",
        "ray": "Ray Serve (for general models)",
    }
    
    for module, name in optional.items():
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            console.print(f"✅ {name}: {version}")
        except ImportError:
            console.print(f"⚠️  {name}: Not installed (optional)")
    
    # GPU check
    console.print("\n[bold]GPU:[/bold]")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            console.print(f"✅ CUDA available: {gpu_count} GPU(s)")
            if verbose:
                for i in range(gpu_count):
                    name = torch.cuda.get_device_name(i)
                    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                    console.print(f"   GPU {i}: {name} ({mem:.1f}GB)")
        else:
            console.print("⚠️  No GPU detected (CPU only)")
    except Exception as e:
        console.print(f"❌ GPU check failed: {e}")
    
    # Config file
    console.print("\n[bold]Configuration:[/bold]")
    config_paths = ["modelium.yaml", "modelium.yml"]
    config_found = False
    for path in config_paths:
        if Path(path).exists():
            console.print(f"✅ Config found: {path}")
            config_found = True
            break
    if not config_found:
        console.print("⚠️  No config file (use 'modelium init')")
    
    # Watch directories
    if config_found:
        try:
            from modelium.config import load_config
            cfg = load_config()
            console.print(f"\n[bold]Watch Directories:[/bold]")
            for dir_path in cfg.orchestration.model_discovery.watch_directories:
                p = Path(dir_path)
                if p.exists():
                    console.print(f"✅ {dir_path}")
                else:
                    console.print(f"⚠️  {dir_path} (doesn't exist)")
        except Exception as e:
            console.print(f"❌ Error loading config: {e}")
    
    # Disk space
    console.print("\n[bold]Disk Space:[/bold]")
    import shutil
    try:
        total, used, free = shutil.disk_usage("/")
        free_gb = free / 1e9
        if free_gb > 50:
            console.print(f"✅ {free_gb:.1f}GB free")
        else:
            console.print(f"⚠️  {free_gb:.1f}GB free (recommend >50GB)")
    except Exception as e:
        console.print(f"❌ Disk check failed: {e}")
    
    # Summary
    console.print()
    if all_ok:
        console.print("[green]✅ All critical dependencies installed[/green]")
        console.print("\nNext steps:")
        console.print("  1. modelium init    # Create config")
        console.print("  2. modelium serve   # Start server")
    else:
        console.print("[red]❌ Some critical dependencies missing[/red]")
        console.print("\nInstall missing dependencies:")
        console.print("  pip install -e .[all]")
        console.print("  pip install vllm  # Separate install")


@app.command()
def init(
    output: Path = typer.Option("modelium.yaml", help="Output config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing file"),
):
    """Initialize a new Modelium configuration file."""
    
    if output.exists() and not force:
        console.print(f"[yellow]Config file already exists: {output}[/yellow]")
        console.print(f"Use --force to overwrite")
        raise typer.Exit(1)
    
    console.print(f"Creating {output}...")
    
    # Copy default config
    import shutil
    from pathlib import Path
    
    # Get package directory
    pkg_dir = Path(__file__).parent.parent
    default_config = pkg_dir / "modelium.yaml"
    
    if default_config.exists():
        shutil.copy(default_config, output)
        console.print(f"[green]✓ Created {output}[/green]")
        console.print(f"\nNext steps:")
        console.print(f"1. Edit {output} with your settings")
        console.print(f"2. Run: modelium serve")
    else:
        console.print(f"[red]Error: Default config not found[/red]")
        raise typer.Exit(1)


@app.command()
def upload(
    file: Path = typer.Argument(..., help="Path to model file"),
    name: Optional[str] = typer.Option(None, help="Model name"),
    framework: Optional[str] = typer.Option(None, help="Framework (pytorch, onnx, tensorflow)"),
    api_url: str = typer.Option("http://localhost:8000", help="Modelium API URL"),
):
    """Upload a model to Modelium."""
    
    if not file.exists():
        console.print(f"[red]Error: File not found: {file}[/red]")
        raise typer.Exit(1)
    
    console.print(f"Uploading {file.name}...")
    
    with open(file, "rb") as f:
        files = {"file": (file.name, f)}
        data = {}
        if name:
            data["name"] = name
        if framework:
            data["framework"] = framework
        
        response = requests.post(f"{api_url}/api/v1/models/upload", files=files, data=data)
    
    if response.status_code == 200:
        model = response.json()
        console.print(f"[green]✓ Upload successful![/green]")
        console.print(f"  Model ID: {model['id']}")
        console.print(f"  Name: {model['name']}")
    else:
        console.print(f"[red]✗ Upload failed: {response.text}[/red]")
        raise typer.Exit(1)


@app.command()
def list(
    api_url: str = typer.Option("http://localhost:8000", help="Modelium API URL"),
):
    """List all models."""
    
    response = requests.get(f"{api_url}/api/v1/models")
    
    if response.status_code != 200:
        console.print(f"[red]Error: {response.text}[/red]")
        raise typer.Exit(1)
    
    models = response.json()
    
    if not models:
        console.print("No models found.")
        return
    
    table = Table(title="Models")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Framework", style="yellow")
    table.add_column("Status", style="magenta")
    table.add_column("Created", style="blue")
    
    for model in models:
        table.add_row(
            model["id"][:12],
            model["name"],
            model.get("framework", "N/A"),
            model["status"],
            model["created_at"][:19],
        )
    
    console.print(table)


@app.command()
def status(
    model_id: str = typer.Argument(..., help="Model ID"),
    api_url: str = typer.Option("http://localhost:8000", help="Modelium API URL"),
):
    """Get model status."""
    
    response = requests.get(f"{api_url}/api/v1/models/{model_id}")
    
    if response.status_code != 200:
        console.print(f"[red]Error: {response.text}[/red]")
        raise typer.Exit(1)
    
    model = response.json()
    
    console.print(f"\n[bold]Model: {model['name']}[/bold]")
    console.print(f"ID: {model['id']}")
    console.print(f"Framework: {model.get('framework', 'N/A')}")
    console.print(f"Status: [{_status_color(model['status'])}]{model['status']}[/]")
    console.print(f"Created: {model['created_at']}")
    
    if model.get("endpoint"):
        console.print(f"Endpoint: {model['endpoint']}")


@app.command()
def watch(
    model_id: str = typer.Argument(..., help="Model ID"),
    api_url: str = typer.Option("http://localhost:8000", help="Modelium API URL"),
):
    """Watch model conversion progress."""
    
    console.print(f"Watching model {model_id}...")
    console.print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            response = requests.get(f"{api_url}/api/v1/models/{model_id}")
            
            if response.status_code != 200:
                console.print(f"[red]Error: {response.text}[/red]")
                break
            
            model = response.json()
            status = model["status"]
            
            console.print(f"[{_status_color(status)}]Status: {status}[/]")
            
            if status in ["deployed", "failed"]:
                console.print(f"\n[bold]Final status: {status}[/bold]")
                if model.get("endpoint"):
                    console.print(f"Endpoint: {model['endpoint']}")
                break
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped watching[/yellow]")


@app.command()
def logs(
    model_id: str = typer.Argument(..., help="Model ID"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow logs"),
    api_url: str = typer.Option("http://localhost:8000", help="Modelium API URL"),
):
    """View model conversion logs."""
    
    try:
        last_count = 0
        
        while True:
            response = requests.get(f"{api_url}/api/v1/models/{model_id}/logs")
            
            if response.status_code != 200:
                console.print(f"[red]Error: {response.text}[/red]")
                break
            
            logs = response.json()
            
            # Print new entries
            entries = logs.get("entries", [])
            for entry in entries[last_count:]:
                console.print(f"[dim]{entry['timestamp']}[/dim] {entry['message']}")
            
            last_count = len(entries)
            
            if not follow or logs.get("status") in ["deployed", "failed"]:
                break
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped following logs[/yellow]")


@app.command()
def delete(
    model_id: str = typer.Argument(..., help="Model ID"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    api_url: str = typer.Option("http://localhost:8000", help="Modelium API URL"),
):
    """Delete a model."""
    
    if not yes:
        confirm = typer.confirm(f"Are you sure you want to delete model {model_id}?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            return
    
    response = requests.delete(f"{api_url}/api/v1/models/{model_id}")
    
    if response.status_code == 200:
        console.print(f"[green]✓ Model deleted[/green]")
    else:
        console.print(f"[red]✗ Delete failed: {response.text}[/red]")
        raise typer.Exit(1)


@app.command()
def deploy(
    model_id: str = typer.Argument(..., help="Model ID"),
    target: str = typer.Option("triton", help="Deployment target (triton, kserve)"),
    replicas: int = typer.Option(3, help="Number of replicas"),
    api_url: str = typer.Option("http://localhost:8000", help="Modelium API URL"),
):
    """Deploy a converted model."""
    
    console.print(f"Deploying model {model_id} to {target}...")
    
    response = requests.post(
        f"{api_url}/api/v1/models/{model_id}/deploy",
        json={"target": target, "replicas": replicas}
    )
    
    if response.status_code == 200:
        result = response.json()
        console.print(f"[green]✓ Deployment started[/green]")
        console.print(f"  Endpoint: {result.get('endpoint')}")
    else:
        console.print(f"[red]✗ Deployment failed: {response.text}[/red]")
        raise typer.Exit(1)


def _status_color(status: str) -> str:
    """Get color for status."""
    colors = {
        "ingesting": "blue",
        "analyzing": "cyan",
        "planning": "cyan",
        "converting": "yellow",
        "deploying": "yellow",
        "deployed": "green",
        "failed": "red",
    }
    return colors.get(status, "white")


if __name__ == "__main__":
    app()

