"""
Runtime Manager - Simplified

Handles ALL runtime interactions (vLLM, Triton, Ray Serve) in ONE place.
No separate connectors/managers - just simple methods to load/unload models.
"""

import logging
import subprocess
import time
import requests
import shutil
import json
from pathlib import Path
from typing import Dict, Optional, List, Any
import psutil

logger = logging.getLogger(__name__)


class RuntimeManager:
    """
    Simple unified runtime manager.
    
    Handles vLLM, Triton, and Ray Serve with straightforward methods.
    User drops model → This loads it → That's it.
    
    Usage:
        manager = RuntimeManager(config)
        manager.load_model("gpt2", "/models/incoming/gpt2", runtime="vllm", gpu=0)
        manager.unload_model("gpt2", runtime="vllm")
    """
    
    def __init__(self, config):
        """
        Initialize runtime manager.
        
        Args:
            config: Modelium configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Track what's loaded
        self._loaded_models: Dict[str, Dict] = {}  # {model_name: {runtime, endpoint, ...}}
        
        # vLLM process management
        self._vllm_processes: Dict[str, subprocess.Popen] = {}
        self._vllm_next_port = 8100
        
        self.logger.info("Runtime Manager initialized")
    
    def load_model(
        self,
        model_name: str,
        model_path: Path,
        runtime: str,
        gpu_id: int = 0,
        settings: Optional[Dict] = None
    ) -> bool:
        """
        Load a model into specified runtime.
        
        Args:
            model_name: Name for the model
            model_path: Path to model files (HuggingFace format)
            runtime: "vllm", "triton", or "ray"
            gpu_id: GPU to use
            settings: Optional runtime-specific settings
        
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"🚀 Loading {model_name} with {runtime}...")
            
            if runtime == "vllm":
                return self._load_vllm(model_name, model_path, gpu_id, settings)
            elif runtime == "triton":
                return self._load_triton(model_name, model_path, gpu_id, settings)
            elif runtime == "ray":
                return self._load_ray(model_name, model_path, gpu_id, settings)
            else:
                self.logger.error(f"Unknown runtime: {runtime}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading {model_name}: {e}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from its runtime.
        
        Args:
            model_name: Name of model to unload
        
        Returns:
            True if successful
        """
        if model_name not in self._loaded_models:
            self.logger.warning(f"{model_name} not loaded")
            return False
        
        try:
            info = self._loaded_models[model_name]
            runtime = info["runtime"]
            
            self.logger.info(f"🛑 Unloading {model_name} from {runtime}...")
            
            if runtime == "vllm":
                return self._unload_vllm(model_name)
            elif runtime == "triton":
                return self._unload_triton(model_name)
            elif runtime == "ray":
                return self._unload_ray(model_name)
            
        except Exception as e:
            self.logger.error(f"Error unloading {model_name}: {e}")
            return False
    
    def is_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        return model_name in self._loaded_models
    
    def get_endpoint(self, model_name: str) -> Optional[str]:
        """Get inference endpoint for a model."""
        if model_name in self._loaded_models:
            return self._loaded_models[model_name].get("endpoint")
        return None
    
    def list_loaded(self) -> List[Dict]:
        """List all loaded models."""
        return [
            {"name": name, **info}
            for name, info in self._loaded_models.items()
        ]
    
    # ==================== vLLM ====================
    
    def _load_vllm(
        self,
        model_name: str,
        model_path: Path,
        gpu_id: int,
        settings: Optional[Dict]
    ) -> bool:
        """Load model by spawning vLLM process."""
        try:
            # Check if vLLM is installed
            try:
                import vllm
            except ImportError:
                self.logger.error(f"   ❌ vLLM not installed!")
                self.logger.error(f"   Install: pip install vllm")
                return False
            
            # Check for Python development headers (required for CUDA compilation)
            # Note: vLLM will compile CUDA code, so we need Python headers
            try:
                import sysconfig
                import sys
                import os
                
                # Get Python version (e.g., "3.11")
                py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
                
                # Check multiple possible locations
                possible_paths = [
                    sysconfig.get_path('include'),  # Standard location
                    f"/usr/include/python{py_version}",  # System location
                    f"/usr/include/python{py_version}m",  # Some distros add 'm'
                    f"/usr/local/include/python{py_version}",  # Local install
                ]
                
                python_h_found = False
                for include_dir in possible_paths:
                    if include_dir:
                        python_h = Path(include_dir) / "Python.h"
                        if python_h.exists():
                            python_h_found = True
                            self.logger.debug(f"   ✅ Found Python.h at {python_h}")
                            break
                
                if not python_h_found:
                    self.logger.warning(f"   ⚠️  Python development headers not found in standard locations")
                    self.logger.warning(f"   Python version: {py_version}")
                    self.logger.warning(f"   Checked: {', '.join(str(p) for p in possible_paths if p)}")
                    self.logger.warning(f"   vLLM requires Python headers to compile CUDA kernels")
                    self.logger.warning(f"   Install: sudo yum install python{py_version.split('.')[0]}-devel  # Amazon Linux")
                    self.logger.warning(f"   Or: sudo apt-get install python{py_version.split('.')[0]}-dev  # Ubuntu/Debian")
                    self.logger.warning(f"   ⚠️  vLLM compilation will likely fail without matching headers")
                    # Don't return False - let vLLM try anyway, it might work
                
                # Check for CUDA toolkit (required for vLLM compilation)
                try:
                    import subprocess
                    nvcc_result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=2)
                    if nvcc_result.returncode == 0:
                        self.logger.debug(f"   ✅ CUDA toolkit found")
                    else:
                        self.logger.warning(f"   ⚠️  CUDA toolkit (nvcc) not found")
                        self.logger.warning(f"   vLLM requires CUDA toolkit for compilation")
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    self.logger.warning(f"   ⚠️  CUDA toolkit (nvcc) not found")
                    self.logger.warning(f"   vLLM requires CUDA toolkit for compilation")
                except Exception:
                    pass  # Ignore other errors
            except Exception as e:
                self.logger.warning(f"   ⚠️  Could not check for Python headers: {e}")
                # Continue anyway - might work
            
            # Validate model path and files
            model_path = Path(model_path).resolve()
            if not model_path.exists():
                self.logger.error(f"   ❌ Model path does not exist: {model_path}")
                return False
            
            config_json = model_path / "config.json"
            if not config_json.exists():
                self.logger.error(f"   ❌ config.json not found at {config_json}")
                return False
            
            # Check for model files
            safetensors_files = list(model_path.glob("*.safetensors"))
            pytorch_files = list(model_path.glob("pytorch_model*.bin"))
            
            if not safetensors_files and not pytorch_files:
                self.logger.error(f"   ❌ No model files found (.safetensors or pytorch_model.bin)")
                self.logger.error(f"   Files in {model_path}: {list(model_path.iterdir())}")
                return False
            
            # Validate safetensors files if present
            if safetensors_files:
                for st_file in safetensors_files:
                    try:
                        # Quick validation: check file size and try to read header
                        file_size = st_file.stat().st_size
                        if file_size == 0:
                            self.logger.error(f"   ❌ {st_file.name} is empty (0 bytes)")
                            return False
                        
                        # Try to validate safetensors file
                        try:
                            from safetensors import safe_open
                            with safe_open(st_file, framework="pt") as f:
                                # Just try to read metadata
                                keys = list(f.keys())[:1]  # Just check first key
                                self.logger.debug(f"   ✅ {st_file.name} is valid safetensors file")
                        except Exception as e:
                            self.logger.error(f"   ❌ {st_file.name} is corrupted or invalid safetensors file")
                            self.logger.error(f"   Error: {e}")
                            self.logger.error(f"   💡 Try re-downloading the model or use a different format")
                            return False
                    except ImportError:
                        self.logger.warning(f"   ⚠️  safetensors library not available, skipping validation")
                    except Exception as e:
                        self.logger.warning(f"   ⚠️  Could not validate {st_file.name}: {e}")
            
            # Allocate port
            port = self._vllm_next_port
            self._vllm_next_port += 1
            
            # Build command
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(model_path),
                "--host", "0.0.0.0",
                "--port", str(port),
                "--dtype", "auto",
            ]
            
            # Set GPU
            env = {
                **subprocess.os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu_id)
            }
            
            self.logger.info(f"")
            self.logger.info(f"=" * 60)
            self.logger.info(f"🚀 SPAWNING vLLM PROCESS")
            self.logger.info(f"   Model: {model_name}")
            self.logger.info(f"   Path: {model_path}")
            self.logger.info(f"   Port: {port}")
            self.logger.info(f"   GPU: {gpu_id}")
            self.logger.info(f"=" * 60)
            
            # Spawn process with stderr redirected to capture errors
            self.logger.info(f"   Command: {' '.join(cmd)}")
            self.logger.info(f"   Environment: CUDA_VISIBLE_DEVICES={gpu_id}")
            
            # Use a named log file in a visible location (not tempfile)
            import os
            log_dir = Path("/tmp/modelium_vllm_logs")
            log_dir.mkdir(exist_ok=True)
            stderr_path = str(log_dir / f"vllm_{model_name}_{port}.log")
            self.logger.info(f"   📝 vLLM stderr will be written to: {stderr_path}")
            
            try:
                self.logger.info(f"   📝 Starting subprocess...")
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=open(stderr_path, 'w'),
                    start_new_session=True,
                    text=True
                )
                
                self.logger.info(f"   ✅ Process spawned! PID: {process.pid}")
                self.logger.info(f"   📋 vLLM stderr log: {stderr_path}")
                self.logger.info(f"   💡 To view logs in real-time: tail -f {stderr_path}")
                self.logger.info(f"   ⏳ Waiting for vLLM to be ready (max 180s)...")
                
                # Wait a moment to see if it crashes immediately
                time.sleep(2)
                if process.poll() is not None:
                    self.logger.error(f"   ❌ Process died immediately with code {process.returncode}")
                    # Read stderr immediately
                    try:
                        if os.path.exists(stderr_path):
                            with open(stderr_path, 'r') as f:
                                stderr_output = f.read()
                                if stderr_output:
                                    self.logger.error(f"   vLLM stderr (immediate failure - full output):")
                                    self.logger.error(f"   {'='*60}")
                                    for line in stderr_output.strip().split('\n'):
                                        if line.strip():
                                            self.logger.error(f"   {line}")
                                    self.logger.error(f"   {'='*60}")
                                    self.logger.error(f"   Full log file: {stderr_path}")
                        else:
                            self.logger.error(f"   ⚠️  Stderr log file not created: {stderr_path}")
                    except Exception as e:
                        self.logger.error(f"   Could not read stderr: {e}")
                    return False
                
                # Wait for ready
                self.logger.info(f"   Process is running, waiting for health check...")
                if self._wait_for_vllm_ready(port, timeout=180, process=process, stderr_path=stderr_path):
                    # Get actual model name from vLLM (might be different from our model_name)
                    vllm_model_name = model_name
                    try:
                        models_resp = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
                        if models_resp.status_code == 200:
                            models_data = models_resp.json()
                            if "data" in models_data and len(models_data["data"]) > 0:
                                vllm_model_name = models_data["data"][0].get("id", model_name)
                                self.logger.info(f"   vLLM model identifier: {vllm_model_name}")
                    except Exception as e:
                        # If we can't query, use model path component as fallback
                        # vLLM often uses the model path or last component
                        vllm_model_name = str(model_path).split("/")[-1] or model_name
                        self.logger.debug(f"   Could not query vLLM models, using fallback: {vllm_model_name}")
                    
                    self._vllm_processes[model_name] = process
                    self._loaded_models[model_name] = {
                        "runtime": "vllm",
                        "endpoint": f"http://localhost:{port}",
                        "port": port,
                        "gpu": gpu_id,
                        "pid": process.pid,
                        "vllm_model_name": vllm_model_name,  # Store actual vLLM model identifier
                    }
                    self.logger.info(f"   ✅ {model_name} ready on port {port} (vLLM name: {vllm_model_name})")
                    # Clean up stderr file on success
                    try:
                        import os
                        os.unlink(stderr_path)
                    except:
                        pass
                    return True
                else:
                    self.logger.error(f"   ❌ vLLM failed to start on port {port}")
                    # Check if process is still running
                    if process.poll() is None:
                        self.logger.error(f"   Process still running but not responding")
                    else:
                        self.logger.error(f"   Process exited with code: {process.returncode}")
                        # Read stderr from file
                        try:
                            with open(stderr_path, 'r') as f:
                                stderr_output = f.read()
                                if stderr_output:
                                    self.logger.error(f"   vLLM stderr output:")
                                    # Show last 100 lines (most recent errors)
                                    lines = stderr_output.strip().split('\n')
                                    for line in lines[-100:]:
                                        if line.strip():
                                            self.logger.error(f"      {line}")
                        except Exception as e:
                            self.logger.error(f"   Could not read stderr file {stderr_path}: {e}")
                    try:
                        process.kill()
                    except:
                        pass
                    finally:
                        # DON'T delete stderr file - keep it for debugging
                        self.logger.error(f"   📋 Stderr log preserved at: {stderr_path}")
                        self.logger.error(f"   💡 View with: cat {stderr_path}")
                    return False
            except Exception as e:
                self.logger.error(f"Error spawning vLLM process: {e}")
                # Clean up stderr file on exception
                try:
                    import os
                    if 'stderr_path' in locals():
                        os.unlink(stderr_path)
                except:
                    pass
                return False
                
        except Exception as e:
            self.logger.error(f"vLLM load failed: {e}")
            return False
    
    def _unload_vllm(self, model_name: str) -> bool:
        """Unload by killing vLLM process."""
        try:
            if model_name not in self._vllm_processes:
                return False
            
            process = self._vllm_processes[model_name]
            
            # Try graceful shutdown
            try:
                proc = psutil.Process(process.pid)
                proc.terminate()
                proc.wait(timeout=10)
            except:
                process.kill()
            
            del self._vllm_processes[model_name]
            del self._loaded_models[model_name]
            
            self.logger.info(f"   ✅ {model_name} unloaded")
            return True
            
        except Exception as e:
            self.logger.error(f"vLLM unload failed: {e}")
            return False
    
    def _wait_for_vllm_ready(self, port: int, timeout: int, process: Optional[subprocess.Popen] = None, stderr_path: Optional[str] = None) -> bool:
        """Wait for vLLM to be ready."""
        start = time.time()
        check_count = 0
        last_stderr_size = 0
        
        while time.time() - start < timeout:
            # Check if process died
            if process and process.poll() is not None:
                self.logger.error(f"")
                self.logger.error(f"   ❌ vLLM PROCESS CRASHED!")
                self.logger.error(f"   Exit code: {process.returncode}")
                self.logger.error(f"   PID: {process.pid}")
                
                # Read stderr from file if available
                if stderr_path:
                    try:
                        import os
                        if os.path.exists(stderr_path):
                            with open(stderr_path, 'r') as f:
                                stderr_output = f.read()
                                if stderr_output:
                                    self.logger.error(f"   {'='*60}")
                                    self.logger.error(f"   vLLM STDERR OUTPUT (full):")
                                    self.logger.error(f"   {'='*60}")
                                    for line in stderr_output.strip().split('\n'):
                                        if line.strip():
                                            self.logger.error(f"   {line}")
                                    self.logger.error(f"   {'='*60}")
                                    self.logger.error(f"   📋 Full log file: {stderr_path}")
                                    self.logger.error(f"   💡 View with: cat {stderr_path}")
                                else:
                                    self.logger.error(f"   ⚠️  Stderr file is empty: {stderr_path}")
                        else:
                            self.logger.error(f"   ⚠️  Stderr file not found: {stderr_path}")
                    except Exception as e:
                        self.logger.error(f"   Could not read stderr: {e}")
                else:
                    self.logger.error(f"   ⚠️  No stderr path provided")
                return False
            
            # Periodically check stderr for errors (even if process is still running)
            if stderr_path and check_count % 5 == 0:  # Every 10 seconds
                try:
                    import os
                    if os.path.exists(stderr_path):
                        current_size = os.path.getsize(stderr_path)
                        if current_size > last_stderr_size:
                            # New content in stderr - check for errors
                            with open(stderr_path, 'r') as f:
                                f.seek(last_stderr_size)
                                new_content = f.read()
                                if new_content:
                                    # Check for common error patterns
                                    error_keywords = ["error", "exception", "traceback", "failed", "fatal"]
                                    for line in new_content.split('\n'):
                                        if any(kw in line.lower() for kw in error_keywords):
                                            self.logger.warning(f"   ⚠️  Error detected in vLLM logs: {line.strip()[:200]}")
                            last_stderr_size = current_size
                except:
                    pass
            
            # Check health endpoint
            try:
                health_url = f"http://localhost:{port}/health"
                resp = requests.get(health_url, timeout=2)
                if resp.status_code == 200:
                    elapsed = time.time() - start
                    self.logger.info(f"   ✅ vLLM ready after {elapsed:.1f}s (health check passed)")
                    return True
                else:
                    self.logger.debug(f"   Health check returned {resp.status_code} (waiting...)")
            except requests.exceptions.ConnectionError as e:
                # Connection refused is normal while starting
                if check_count % 10 == 0:  # Only log every 20 seconds
                    self.logger.debug(f"   Health endpoint not ready yet (connection refused is normal)")
            except requests.exceptions.RequestException as e:
                if check_count % 10 == 0:  # Only log every 20 seconds
                    self.logger.debug(f"   Health check error: {type(e).__name__}")
            
            check_count += 1
            if check_count % 10 == 0:  # Log every 20 seconds
                elapsed = time.time() - start
                self.logger.info(f"   Still waiting... ({elapsed:.0f}s elapsed, PID: {process.pid if process else 'N/A'})")
                if stderr_path:
                    self.logger.info(f"   💡 Monitor logs: tail -f {stderr_path}")
            
            time.sleep(2)
        
        self.logger.error(f"   Timeout after {timeout}s")
        if stderr_path:
            self.logger.error(f"   📋 Check logs: {stderr_path}")
        return False
    
    # ==================== Triton ====================
    
    def _load_triton(
        self,
        model_name: str,
        model_path: Path,
        gpu_id: int,
        settings: Optional[Dict]
    ) -> bool:
        """Load model via Triton API."""
        try:
            endpoint = self.config.triton.endpoint
            
            # Check Triton is running
            resp = requests.get(f"{endpoint}/v2/health/ready", timeout=5)
            if resp.status_code != 200:
                self.logger.error("Triton not ready")
                return False
            
            # Triton needs specific structure - create it
            triton_model_dir = Path("/models/triton-repository") / model_name
            if not triton_model_dir.exists():
                triton_model_dir.mkdir(parents=True)
                version_dir = triton_model_dir / "1"
                version_dir.mkdir()
                
                # Copy model files
                for file in model_path.iterdir():
                    if file.is_file():
                        shutil.copy2(file, version_dir / file.name)
                
                # Create basic config.pbtxt
                config_content = f'''
name: "{model_name}"
backend: "pytorch"
max_batch_size: 32
'''
                (triton_model_dir / "config.pbtxt").write_text(config_content)
            
            # Call Triton load API
            self.logger.info(f"   Calling Triton load API...")
            resp = requests.post(
                f"{endpoint}/v2/repository/models/{model_name}/load",
                timeout=180
            )
            
            if resp.status_code == 200:
                self._loaded_models[model_name] = {
                    "runtime": "triton",
                    "endpoint": endpoint,
                    "gpu": gpu_id,
                }
                self.logger.info(f"   ✅ {model_name} loaded in Triton")
                return True
            else:
                self.logger.error(f"   ❌ Triton load failed: {resp.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Triton load failed: {e}")
            return False
    
    def _unload_triton(self, model_name: str) -> bool:
        """Unload from Triton via API."""
        try:
            endpoint = self.config.triton.endpoint
            resp = requests.post(
                f"{endpoint}/v2/repository/models/{model_name}/unload",
                timeout=30
            )
            
            if resp.status_code == 200:
                del self._loaded_models[model_name]
                self.logger.info(f"   ✅ {model_name} unloaded from Triton")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Triton unload failed: {e}")
            return False
    
    # ==================== Ray Serve ====================
    
    def _load_ray(
        self,
        model_name: str,
        model_path: Path,
        gpu_id: int,
        settings: Optional[Dict]
    ) -> bool:
        """Load model via Ray Serve."""
        try:
            # Check if Ray is available
            try:
                import ray
                from ray import serve
            except ImportError:
                self.logger.error("Ray not installed (pip install ray[serve])")
                return False
            
            # Initialize Ray if needed
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            
            try:
                serve.start(detached=True)
            except:
                pass
            
            # Simple deployment
            @serve.deployment(
                name=model_name,
                ray_actor_options={"num_gpus": 1, "num_cpus": 2},
                num_replicas=1,
            )
            class Model:
                def __init__(self, path):
                    self.path = path
                
                def __call__(self, request):
                    return {"model": self.path, "status": "ok"}
            
            deployment = Model.bind(str(model_path))
            serve.run(deployment, name=model_name, route_prefix=f"/{model_name}")
            
            self._loaded_models[model_name] = {
                "runtime": "ray",
                "endpoint": f"http://localhost:8002/{model_name}",
                "gpu": gpu_id,
            }
            
            self.logger.info(f"   ✅ {model_name} deployed via Ray")
            return True
            
        except Exception as e:
            self.logger.error(f"Ray load failed: {e}")
            return False
    
    def _unload_ray(self, model_name: str) -> bool:
        """Unload from Ray Serve."""
        try:
            from ray import serve
            serve.delete(model_name)
            del self._loaded_models[model_name]
            self.logger.info(f"   ✅ {model_name} undeployed from Ray")
            return True
        except Exception as e:
            self.logger.error(f"Ray unload failed: {e}")
            return False
    
    # ==================== Inference ====================
    
    def inference(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference on a loaded model.
        
        Routes to correct runtime automatically.
        """
        if model_name not in self._loaded_models:
            return {"error": "Model not loaded"}
        
        info = self._loaded_models[model_name]
        runtime = info["runtime"]
        
        try:
            if runtime == "vllm":
                # vLLM uses OpenAI-compatible API
                endpoint = info['endpoint']
                
                # Get actual model name from vLLM (might be different from our model_name)
                # vLLM might use the model path or a derived name
                actual_model_name = info.get("vllm_model_name", model_name)
                
                # ALWAYS query vLLM's /v1/models to get the exact model identifier
                # This ensures we use the correct name that vLLM expects
                try:
                    self.logger.info(f"   🔍 Querying vLLM /v1/models to get exact model identifier...")
                    models_resp = requests.get(f"{endpoint}/v1/models", timeout=5)
                    if models_resp.status_code == 200:
                        models_data = models_resp.json()
                        self.logger.debug(f"   vLLM /v1/models response: {models_data}")
                        if "data" in models_data and len(models_data["data"]) > 0:
                            # Use the first available model's id (this is what vLLM expects)
                            model_info = models_data["data"][0]
                            actual_model_name = model_info.get("id", model_name)
                            
                            # Check what tasks this model supports
                            owned_by = model_info.get("owned_by", "")
                            self.logger.info(f"   ✅ Using vLLM model identifier: {actual_model_name}")
                            self.logger.info(f"   📋 Model info: owned_by={owned_by}, id={actual_model_name}")
                            
                            # Log full model info for debugging
                            self.logger.debug(f"   Full model info: {model_info}")
                        else:
                            self.logger.warning(f"   ⚠️  vLLM /v1/models returned no models, using stored: {actual_model_name}")
                    else:
                        self.logger.warning(f"   ⚠️  vLLM /v1/models returned {models_resp.status_code}, using stored: {actual_model_name}")
                except Exception as e:
                    self.logger.warning(f"   ⚠️  Could not query vLLM /v1/models: {e}, using stored: {actual_model_name}")
                
                # Try /v1/chat/completions first (preferred in vLLM 0.10+)
                chat_completions_tried = False
                try:
                    self.logger.debug(f"   Trying Chat Completions API with model: {actual_model_name}")
                    resp = requests.post(
                        f"{endpoint}/v1/chat/completions",
                        json={
                            "model": actual_model_name,
                            "messages": [
                                {"role": "user", "content": prompt}
                            ],
                            "max_tokens": max_tokens,
                            "temperature": kwargs.get("temperature", 0.7),
                        },
                        timeout=30
                    )
                    chat_completions_tried = True
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        # Extract text from chat format
                        if "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0].get("message", {}).get("content", "")
                            self.logger.debug(f"   ✅ Chat Completions API succeeded")
                            return {
                                "text": content,
                                "choices": result.get("choices", []),
                                "usage": result.get("usage", {})
                            }
                    else:
                        # Read error message
                        try:
                            error_data = resp.json()
                            error_msg = error_data.get("error", {}).get("message", resp.text)
                        except:
                            error_msg = resp.text
                        
                        self.logger.debug(f"   Chat Completions returned {resp.status_code}: {error_msg}")
                        
                        if resp.status_code == 400 and "does not support" in error_msg.lower():
                            # Model doesn't support chat completions, try legacy
                            self.logger.debug(f"   Model doesn't support Chat Completions, trying legacy Completions API")
                        else:
                            resp.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    error_msg = str(e)
                    self.logger.debug(f"   Chat Completions HTTP error: {error_msg}")
                    if "does not support" in error_msg.lower() or "400" in error_msg:
                        # Fall back to legacy completions API
                        self.logger.debug(f"   Will try legacy Completions API")
                    else:
                        raise
                except Exception as e:
                    self.logger.debug(f"   Chat Completions exception: {e}")
                    if not chat_completions_tried:
                        raise
                
                # Fallback to legacy /v1/completions API
                self.logger.debug(f"   Trying legacy Completions API with model: {actual_model_name}")
                try:
                    resp = requests.post(
                        f"{endpoint}/v1/completions",
                        json={
                            "model": actual_model_name,
                            "prompt": prompt,
                            "max_tokens": max_tokens,
                            "temperature": kwargs.get("temperature", 0.7),
                            **{k: v for k, v in kwargs.items() if k != "temperature"}
                        },
                        timeout=30
                    )
                    
                    if resp.status_code == 200:
                        self.logger.debug(f"   ✅ Legacy Completions API succeeded")
                        return resp.json()
                    else:
                        # Read error message
                        try:
                            error_data = resp.json()
                            error_msg = error_data.get("error", {}).get("message", resp.text)
                        except:
                            error_msg = resp.text
                        
                        self.logger.error(f"   ❌ Completions API failed: {error_msg}")
                        resp.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    # Try to get better error message
                    try:
                        error_data = e.response.json()
                        error_msg = error_data.get("error", {}).get("message", str(e))
                    except:
                        error_msg = str(e)
                    
                    self.logger.error(f"   ❌ Both Chat and Completions APIs failed")
                    self.logger.error(f"   Model: {actual_model_name}")
                    self.logger.error(f"   Endpoint: {endpoint}")
                    self.logger.error(f"   Error: {error_msg}")
                    
                    # Try to get more info about what the model supports
                    try:
                        models_resp = requests.get(f"{endpoint}/v1/models", timeout=5)
                        if models_resp.status_code == 200:
                            models_data = models_resp.json()
                            self.logger.error(f"   📋 Available models: {models_data}")
                    except:
                        pass
                    
                    # Return error in a format the API can handle
                    return {
                        "error": f"vLLM inference failed: {error_msg}",
                        "model": actual_model_name,
                        "endpoint": endpoint,
                        "suggestion": "Check vLLM logs at /tmp/modelium_vllm_logs/ for details"
                    }
            
            elif runtime == "triton":
                # Triton uses KServe v2 (simplified example)
                resp = requests.post(
                    f"{info['endpoint']}/v2/models/{model_name}/infer",
                    json={"inputs": [{"data": [prompt]}]},
                    timeout=30
                )
                return resp.json()
            
            elif runtime == "ray":
                # Ray custom endpoint
                resp = requests.post(
                    info['endpoint'],
                    json={"prompt": prompt},
                    timeout=30
                )
                return resp.json()
            
        except Exception as e:
            return {"error": str(e)}

