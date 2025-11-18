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
            
            # Build command - vLLM 0.11+ uses CLI: vllm serve
            # Try new CLI first, fallback to old module path
            cmd = [
                "vllm", "serve",
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
                # Try new vLLM CLI first (vLLM 0.11+)
                try:
                    process = subprocess.Popen(
                        cmd,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=open(stderr_path, 'w'),
                        start_new_session=True,
                        text=True
                    )
                    self.logger.info(f"   ✅ Using vLLM CLI: vllm serve")
                except FileNotFoundError:
                    # Fallback to old module path if vllm CLI not found (vLLM 0.10-)
                    self.logger.warning(f"   ⚠️  'vllm' CLI not found, trying old module path...")
                    cmd_old = [
                        "python", "-m", "vllm.entrypoints.openai.api_server",
                        "--model", str(model_path),
                        "--host", "0.0.0.0",
                        "--port", str(port),
                        "--dtype", "auto",
                    ]
                    process = subprocess.Popen(
                        cmd_old,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=open(stderr_path, 'w'),
                        start_new_session=True,
                        text=True
                    )
                    self.logger.info(f"   ✅ Using old module path (python -m vllm.entrypoints.openai.api_server)")
                
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
                                model_info = models_data["data"][0]
                                vllm_model_name = model_info.get("id", model_name)
                                
                                # For GPT-2 and similar models, vLLM might use the path
                                # Try to extract just the model name if it's a path
                                if "/" in vllm_model_name:
                                    # If it's a path, try using just the last component
                                    # But also store the full path as fallback
                                    path_components = vllm_model_name.split("/")
                                    simple_name = path_components[-1] if path_components else model_name
                                    self.logger.info(f"   vLLM model identifier: {vllm_model_name}")
                                    self.logger.info(f"   Will try both: '{vllm_model_name}' and '{simple_name}'")
                                    # Store both for inference to try
                                    vllm_model_name = vllm_model_name  # Keep full path as primary
                                else:
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
                self.logger.error("   ❌ Ray Serve not installed!")
                self.logger.error("   💡 Install: pip install 'ray[serve]'")
                self.logger.error("   💡 Or disable ray_serve.enabled in modelium.yaml")
                return False
            
            # Initialize Ray if needed
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            
            # Check if Ray Serve is already running and clean up if needed
            try:
                # Try to get existing deployments
                existing_deployments = serve.list_deployments()
                if model_name in existing_deployments:
                    self.logger.info(f"   🧹 Cleaning up existing deployment: {model_name}")
                    serve.delete(model_name)
                    time.sleep(2)  # Give it time to clean up
            except:
                pass
            
            # Start Ray Serve (or connect if already running)
            try:
                # Check if serve is already running
                try:
                    serve.get_deployment(model_name)
                    self.logger.warning(f"   ⚠️  Deployment {model_name} already exists, deleting...")
                    serve.delete(model_name)
                    time.sleep(2)
                except:
                    pass
                
                # Start serve with explicit HTTP options to avoid port conflicts
                try:
                    serve.start(
                        detached=True,
                        http_options={
                            "host": "0.0.0.0",
                            "port": 8002,
                            "location": "EveryNode"
                        }
                    )
                except Exception as e:
                    # If already started, that's OK
                    if "already running" not in str(e).lower() and "address already in use" not in str(e).lower():
                        self.logger.warning(f"   ⚠️  Ray Serve start warning: {e}")
            except Exception as e:
                self.logger.warning(f"   ⚠️  Ray Serve may already be running: {e}")
            
            # GPT-2 deployment with actual model loading
            # Ray Serve will automatically assign a GPU when num_gpus > 0
            # We use CUDA_VISIBLE_DEVICES to limit which GPU Ray can see
            import torch
            import os
            ray_env = {}
            if gpu_id >= 0:
                try:
                    if torch.cuda.is_available():
                        # Set CUDA_VISIBLE_DEVICES so Ray only sees the GPU we want
                        ray_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                        self.logger.info(f"   Setting CUDA_VISIBLE_DEVICES={gpu_id} for Ray actor")
                except:
                    pass
            
            @serve.deployment(
                name=model_name,
                ray_actor_options={
                    "num_gpus": 1 if gpu_id >= 0 else 0,
                    "num_cpus": 2,
                    "runtime_env": {"env_vars": ray_env} if ray_env else {}
                },
                num_replicas=1,
            )
            class GPT2Model:
                def __init__(self, model_path: str):
                    import torch
                    from transformers import GPT2LMHeadModel, GPT2Tokenizer
                    import os
                    
                    self.logger = logging.getLogger(f"RayServe.{model_name}")
                    self.logger.info(f"Loading GPT-2 model from {model_path}...")
                    
                    # Ray Serve automatically assigns GPU when num_gpus > 0
                    # Use the first available CUDA device (Ray handles assignment)
                    if torch.cuda.is_available():
                        # Ray Serve assigns GPU automatically, use cuda:0 (Ray's assigned GPU)
                        self.device = "cuda:0"
                        self.logger.info(f"Using GPU (Ray-assigned): {self.device}")
                        self.logger.info(f"Available GPUs: {torch.cuda.device_count()}")
                    else:
                        self.device = "cpu"
                        self.logger.info(f"Using CPU (no GPU available)")
                    
                    # Load model and tokenizer
                    try:
                        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
                        self.model = GPT2LMHeadModel.from_pretrained(model_path)
                        self.model.to(self.device)
                        self.model.eval()
                        self.logger.info(f"✅ GPT-2 model loaded successfully on {self.device}")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to load model: {e}")
                        import traceback
                        self.logger.error(f"Traceback: {traceback.format_exc()}")
                        raise
                
                def __call__(self, request):
                    import torch
                    
                    # Ray Serve can pass request as dict or Starlette Request object
                    # Handle both cases
                    if hasattr(request, "json"):  # Starlette Request object
                        # For sync deployments, Ray Serve should pass dict, but handle Request object
                        try:
                            # Try to get JSON body
                            if hasattr(request, "_json"):
                                request_dict = request._json
                            elif hasattr(request, "json"):
                                # If it's a coroutine, we can't await it in sync context
                                # Ray Serve should handle this, but let's try to get it
                                import json
                                if hasattr(request, "body"):
                                    request_dict = json.loads(request.body.decode())
                                else:
                                    request_dict = {}
                            else:
                                request_dict = {}
                        except:
                            request_dict = {}
                    elif isinstance(request, dict):
                        request_dict = request
                    else:
                        # Try to convert to dict
                        try:
                            request_dict = dict(request) if hasattr(request, "__iter__") and not isinstance(request, str) else {}
                        except:
                            request_dict = {}
                    
                    prompt = request_dict.get("prompt", "") if isinstance(request_dict, dict) else ""
                    max_tokens = request_dict.get("max_tokens", 100) if isinstance(request_dict, dict) else 100
                    temperature = request_dict.get("temperature", 0.7) if isinstance(request_dict, dict) else 0.7
                    
                    self.logger.debug(f"Received request type: {type(request)}")
                    self.logger.debug(f"Request dict type: {type(request_dict)}")
                    self.logger.debug(f"Request dict keys: {list(request_dict.keys()) if isinstance(request_dict, dict) else 'not a dict'}")
                    self.logger.debug(f"Prompt: {prompt[:50] if prompt else 'None'}..., max_tokens={max_tokens}, temp={temperature}")
                    
                    if not prompt:
                        self.logger.error(f"Prompt is required but not found in request")
                        self.logger.error(f"Request type: {type(request)}")
                        self.logger.error(f"Request dict: {request_dict}")
                        return {"error": "Prompt is required", "received_keys": list(request_dict.keys()) if isinstance(request_dict, dict) else [], "request_type": str(type(request))}
                    
                    try:
                        # Tokenize input
                        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                        
                        if inputs.shape[1] == 0:
                            return {"error": "Empty input after tokenization"}
                        
                        # Generate
                        with torch.no_grad():
                            outputs = self.model.generate(
                                inputs,
                                max_length=inputs.shape[1] + max_tokens,
                                min_length=inputs.shape[1] + 1,  # At least generate 1 new token
                                temperature=temperature,
                                do_sample=True,
                                pad_token_id=self.tokenizer.eos_token_id,
                                eos_token_id=self.tokenizer.eos_token_id
                            )
                        
                        # Check if we got output
                        if outputs.shape[0] == 0 or outputs.shape[1] == 0:
                            return {"error": "Model generated empty output"}
                        
                        # Decode output
                        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        # Extract only the new tokens (remove prompt)
                        # Handle case where generated_text might be shorter than prompt
                        if len(generated_text) > len(prompt):
                            new_text = generated_text[len(prompt):].strip()
                        else:
                            # If generated text is same or shorter, return it all
                            new_text = generated_text.strip()
                        
                        return {
                            "text": new_text,
                            "full_text": generated_text,
                            "model": model_name
                        }
                    except Exception as e:
                        import traceback
                        self.logger.error(f"Inference error: {e}")
                        self.logger.error(f"Traceback: {traceback.format_exc()}")
                        return {"error": str(e), "error_type": type(e).__name__}
            
            deployment = GPT2Model.bind(str(model_path))
            serve.run(deployment, name=model_name, route_prefix=f"/{model_name}")
            
            endpoint = f"http://localhost:8002/{model_name}"
            self._loaded_models[model_name] = {
                "runtime": "ray",
                "endpoint": endpoint,
                "gpu": gpu_id,
                "model_path": str(model_path),
            }
            
            # Wait for Ray Serve to be ready
            self.logger.info(f"   ⏳ Waiting for Ray Serve deployment to be ready...")
            if self._wait_for_ray_ready(endpoint, timeout=300):
                self.logger.info(f"   ✅ {model_name} deployed and ready via Ray Serve")
                return True
            else:
                self.logger.error(f"   ❌ Ray Serve deployment not ready after timeout")
                # Clean up
                try:
                    serve.delete(model_name)
                except:
                    pass
                del self._loaded_models[model_name]
                return False
            
        except Exception as e:
            self.logger.error(f"Ray load failed: {e}")
            return False
    
    def _wait_for_ray_ready(self, endpoint: str, timeout: int = 300) -> bool:
        """Wait for Ray Serve deployment to be ready."""
        import time
        start = time.time()
        check_count = 0
        
        while time.time() - start < timeout:
            check_count += 1
            try:
                # Try to hit the endpoint
                resp = requests.get(endpoint, timeout=2)
                if resp.status_code in [200, 404]:  # 404 is OK, means endpoint exists
                    elapsed = time.time() - start
                    self.logger.info(f"   ✅ Ray Serve ready after {elapsed:.1f}s")
                    return True
            except requests.exceptions.ConnectionError:
                # Connection refused is normal while starting
                if check_count % 10 == 0:  # Log every 20 seconds
                    elapsed = time.time() - start
                    self.logger.debug(f"   Ray Serve not ready yet ({elapsed:.0f}s elapsed)...")
            except Exception as e:
                # Other errors might mean it's starting
                if check_count % 10 == 0:
                    self.logger.debug(f"   Ray Serve check error (normal during startup): {e}")
            
            time.sleep(2)
        
        self.logger.error(f"   ❌ Ray Serve not ready after {timeout}s")
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
        try:
            self.logger.info(f"🔍 Starting inference for model: {model_name}")
            self.logger.debug(f"   Prompt: {prompt[:100]}...")
            self.logger.debug(f"   Max tokens: {max_tokens}, kwargs: {kwargs}")
            
            if model_name not in self._loaded_models:
                self.logger.error(f"   ❌ Model '{model_name}' not in loaded models")
                self.logger.debug(f"   Available models: {list(self._loaded_models.keys())}")
                return {"error": "Model not loaded"}
            
            info = self._loaded_models[model_name]
            runtime = info.get("runtime")
            self.logger.debug(f"   Runtime: {runtime}")
            self.logger.debug(f"   Model info: {info}")
            
            if not runtime:
                self.logger.error(f"   ❌ No runtime specified for model '{model_name}'")
                return {"error": "No runtime specified"}
            
            if runtime == "vllm":
                # vLLM uses OpenAI-compatible API
                endpoint = info.get('endpoint')
                self.logger.debug(f"   Endpoint from info: {endpoint} (type: {type(endpoint)})")
                
                if endpoint is None:
                    self.logger.error(f"   ❌ No endpoint found in model info!")
                    if isinstance(info, dict):
                        import builtins
                        info_keys = builtins.list(info.keys())
                        self.logger.error(f"   Model info keys: {info_keys}")
                    else:
                        self.logger.error(f"   Model info is not a dict: {type(info)}")
                    self.logger.error(f"   Full model info: {info}")
                    return {"error": "No endpoint configured for vLLM model"}
                
                if not isinstance(endpoint, str):
                    self.logger.error(f"   ❌ Endpoint is not a string: {endpoint} (type: {type(endpoint)})")
                    return {"error": f"Invalid endpoint type: {type(endpoint)}"}
                
                # Get actual model name from vLLM (might be different from our model_name)
                # vLLM might use the model path or a derived name
                self.logger.debug(f"   Getting model name from info dict...")
                actual_model_name = info.get("vllm_model_name")
                self.logger.debug(f"   vllm_model_name from info: {actual_model_name} (type: {type(actual_model_name)})")
                
                if actual_model_name is None:
                    actual_model_name = model_name
                    self.logger.debug(f"   Using model_name as fallback: {actual_model_name}")
                
                # ALWAYS query vLLM's /v1/models to get the exact model identifier
                # This ensures we use the correct name that vLLM expects
                try:
                    self.logger.info(f"   🔍 Querying vLLM /v1/models to get exact model identifier...")
                    self.logger.debug(f"   Endpoint: {endpoint}")
                    models_resp = requests.get(f"{endpoint}/v1/models", timeout=5)
                    self.logger.debug(f"   Response status: {models_resp.status_code}")
                    
                    if models_resp.status_code == 200:
                        models_data = models_resp.json()
                        self.logger.debug(f"   vLLM /v1/models response: {models_data}")
                        
                        if "data" in models_data and len(models_data["data"]) > 0:
                            # Use the first available model's id (this is what vLLM expects)
                            model_info = models_data["data"][0]
                            new_model_name = model_info.get("id")
                            self.logger.debug(f"   Model id from response: {new_model_name} (type: {type(new_model_name)})")
                            
                            if new_model_name is not None:
                                actual_model_name = new_model_name
                            
                            # Check what tasks this model supports
                            owned_by = model_info.get("owned_by", "")
                            self.logger.info(f"   ✅ Using vLLM model identifier: {actual_model_name}")
                            self.logger.info(f"   📋 Model info: owned_by={owned_by}, id={actual_model_name}")
                            
                            # For GPT-2, if model name is a path, also try just the model name
                            # vLLM might accept either format
                            try:
                                if actual_model_name and isinstance(actual_model_name, str):
                                    if "/" in actual_model_name and "gpt2" in actual_model_name.lower():
                                        # Extract just "gpt2" from path
                                        path_parts = actual_model_name.split("/")
                                        simple_name = [p for p in path_parts if "gpt2" in p.lower()][-1] if path_parts else None
                                        if simple_name:
                                            self.logger.info(f"   💡 GPT-2 detected, will also try simple name: '{simple_name}'")
                            except Exception as e:
                                self.logger.warning(f"   ⚠️  Error checking GPT-2 name format: {e}")
                            
                            # Log full model info for debugging
                            self.logger.debug(f"   Full model info: {model_info}")
                        else:
                            self.logger.warning(f"   ⚠️  vLLM /v1/models returned no models, using stored: {actual_model_name}")
                    else:
                        self.logger.warning(f"   ⚠️  vLLM /v1/models returned {models_resp.status_code}, using stored: {actual_model_name}")
                except Exception as e:
                    self.logger.warning(f"   ⚠️  Could not query vLLM /v1/models: {e}")
                    self.logger.debug(f"   Exception type: {type(e)}, args: {e.args}")
                    import traceback
                    self.logger.debug(f"   Traceback: {traceback.format_exc()}")
                    self.logger.debug(f"   Using stored: {actual_model_name}")
                
                # For GPT-2, try different model name formats
                # vLLM might accept the full path or just "gpt2"
                self.logger.debug(f"   Preparing model names to try...")
                self.logger.debug(f"   actual_model_name: {actual_model_name} (type: {type(actual_model_name)})")
                
                if actual_model_name is None:
                    actual_model_name = model_name
                    self.logger.warning(f"   ⚠️  actual_model_name was None, using model_name: {model_name}")
                
                if not isinstance(actual_model_name, str):
                    self.logger.error(f"   ❌ actual_model_name is not a string: {actual_model_name} (type: {type(actual_model_name)})")
                    actual_model_name = str(actual_model_name) if actual_model_name else model_name
                    self.logger.warning(f"   Converted to string: {actual_model_name}")
                
                model_names_to_try = [actual_model_name]
                
                try:
                    if actual_model_name and isinstance(actual_model_name, str):
                        if "/" in actual_model_name and "gpt2" in actual_model_name.lower():
                            # Extract just "gpt2" from path
                            path_parts = actual_model_name.split("/")
                            simple_name = [p for p in path_parts if "gpt2" in p.lower()][-1] if path_parts else None
                            if simple_name and simple_name != actual_model_name:
                                model_names_to_try.append(simple_name)
                                self.logger.info(f"   💡 Will try model names: {model_names_to_try}")
                except Exception as e:
                    self.logger.error(f"   ❌ Error preparing model name variants: {e}")
                    self.logger.debug(f"   Exception type: {type(e)}, args: {e.args}")
                    import traceback
                    self.logger.debug(f"   Traceback: {traceback.format_exc()}")
                    # Continue with just the original name
                
                # Try /v1/chat/completions first (preferred in vLLM 0.10+)
                chat_completions_tried = False
                chat_success = False
                for model_name_variant in model_names_to_try:
                    try:
                        self.logger.debug(f"   Trying Chat Completions API with model: {model_name_variant}")
                        resp = requests.post(
                            f"{endpoint}/v1/chat/completions",
                            json={
                                "model": model_name_variant,
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
                            self.logger.debug(f"   Chat Completions response: {result}")
                            
                            # CRITICAL: vLLM can return 200 with an error object!
                            if "error" in result:
                                error_msg = result.get("error", {}).get("message", "Unknown error") if isinstance(result.get("error"), dict) else str(result.get("error"))
                                self.logger.warning(f"   ⚠️  Chat Completions returned 200 but with error: {error_msg}")
                                # Try next variant or fallback
                                continue
                            
                            # Extract text from chat format
                            if "choices" in result and len(result["choices"]) > 0:
                                content = result["choices"][0].get("message", {}).get("content", "")
                                self.logger.info(f"   ✅ Chat Completions API succeeded with model: {model_name_variant}")
                                self.logger.debug(f"   Generated content: {content[:100]}...")
                                return {
                                    "text": content,
                                    "choices": result.get("choices", []),
                                    "usage": result.get("usage", {})
                                }
                            else:
                                # Response is 200 but no choices - log and try next variant or fallback
                                self.logger.warning(f"   ⚠️  Chat Completions returned 200 but no choices in response")
                                if isinstance(result, dict):
                                    import builtins
                                    result_keys = builtins.list(result.keys())
                                    self.logger.debug(f"   Response keys: {result_keys}")
                                else:
                                    self.logger.debug(f"   Response is not a dict: {type(result)}")
                                self.logger.debug(f"   Full response: {result}")
                                # Don't break - try next model name variant or fallback to completions
                                continue
                        else:
                            # Read error message
                            try:
                                error_data = resp.json()
                                error_msg = error_data.get("error", {}).get("message", resp.text)
                            except:
                                error_msg = resp.text or ""
                            
                            # Ensure error_msg is a string
                            if error_msg is None:
                                error_msg = ""
                            
                            self.logger.debug(f"   Chat Completions with '{model_name_variant}' returned {resp.status_code}: {error_msg}")
                            
                            # If this model name doesn't work, try next variant
                            if resp.status_code == 400 and error_msg and "does not support" not in error_msg.lower():
                                continue  # Try next model name variant
                            elif resp.status_code == 400 and error_msg and "does not support" in error_msg.lower():
                                # Model doesn't support chat completions at all, break and try completions
                                break
                    except requests.exceptions.HTTPError as e:
                        error_msg = str(e) or ""
                        self.logger.debug(f"   Chat Completions HTTP error with '{model_name_variant}': {error_msg}")
                        if error_msg and "does not support" not in error_msg.lower():
                            continue  # Try next model name variant
                        break
                    except Exception as e:
                        self.logger.debug(f"   Chat Completions exception with '{model_name_variant}': {e}")
                        continue  # Try next model name variant
                
                # Note: chat_success is no longer used since we return immediately on success
                # If we get here, Chat Completions didn't work, try Completions API
                
                # Fallback to legacy /v1/completions API
                # Try all model name variants
                for model_name_variant in model_names_to_try:
                    try:
                        self.logger.debug(f"   Trying legacy Completions API with model: {model_name_variant}")
                        resp = requests.post(
                            f"{endpoint}/v1/completions",
                            json={
                                "model": model_name_variant,
                                "prompt": prompt,
                                "max_tokens": max_tokens,
                                "temperature": kwargs.get("temperature", 0.7),
                                **{k: v for k, v in kwargs.items() if k != "temperature"}
                            },
                            timeout=30
                        )
                        
                        if resp.status_code == 200:
                            result = resp.json()
                            self.logger.debug(f"   Completions API response: {result}")
                            
                            # CRITICAL: vLLM can return 200 with an error object!
                            if "error" in result:
                                error_msg = result.get("error", {}).get("message", "Unknown error") if isinstance(result.get("error"), dict) else str(result.get("error"))
                                self.logger.warning(f"   ⚠️  Completions API returned 200 but with error: {error_msg}")
                                # Try next variant
                                continue
                            
                            # Check for choices/text in response
                            if "choices" in result and len(result["choices"]) > 0:
                                self.logger.info(f"   ✅ Legacy Completions API succeeded with model: {model_name_variant}")
                                return result
                            elif "text" in result:
                                self.logger.info(f"   ✅ Legacy Completions API succeeded with model: {model_name_variant}")
                                return result
                            else:
                                self.logger.warning(f"   ⚠️  Completions API returned 200 but no choices/text in response")
                                if isinstance(result, dict):
                                    import builtins
                                    result_keys = builtins.list(result.keys())
                                    self.logger.debug(f"   Response keys: {result_keys}")
                                else:
                                    self.logger.debug(f"   Response is not a dict: {type(result)}")
                                # Try next variant
                                continue
                        else:
                            # Read error message
                            try:
                                error_data = resp.json()
                                error_msg = error_data.get("error", {}).get("message", resp.text)
                            except:
                                error_msg = resp.text or ""
                            
                            # Ensure error_msg is a string
                            if error_msg is None:
                                error_msg = ""
                            
                            self.logger.debug(f"   Completions API with '{model_name_variant}' returned {resp.status_code}: {error_msg}")
                            
                            # If this model name doesn't work, try next variant
                            if resp.status_code == 400 and error_msg and "does not support" not in error_msg.lower():
                                continue  # Try next model name variant
                            elif resp.status_code == 400 and error_msg and "does not support" in error_msg.lower():
                                # Model doesn't support completions at all
                                self.logger.error(f"   ❌ Model '{model_name_variant}' does not support Completions API")
                                break
                    except requests.exceptions.HTTPError as e:
                        # Try to get better error message
                        try:
                            error_data = e.response.json()
                            error_msg = error_data.get("error", {}).get("message", str(e))
                        except:
                            error_msg = str(e) or ""
                        
                        # Ensure error_msg is a string
                        if error_msg is None:
                            error_msg = ""
                        
                        self.logger.debug(f"   Completions HTTP error with '{model_name_variant}': {error_msg}")
                        if error_msg and "does not support" not in error_msg.lower():
                            continue  # Try next model name variant
                        # If we get here, all variants failed
                        if model_name_variant == model_names_to_try[-1]:
                            # Last variant, raise the error
                            raise
                    except Exception as e:
                        self.logger.debug(f"   Completions exception with '{model_name_variant}': {e}")
                        if model_name_variant == model_names_to_try[-1]:
                            # Last variant, re-raise
                            raise
                        continue  # Try next variant
                
                # If we get here, all model name variants failed for both APIs
                self.logger.error(f"   ❌ Both Chat and Completions APIs failed for all model name variants")
                self.logger.error(f"   Tried model names: {model_names_to_try}")
                self.logger.error(f"   Endpoint: {endpoint}")
                
                # Try to get more info about what the model supports
                try:
                    models_resp = requests.get(f"{endpoint}/v1/models", timeout=5)
                    if models_resp.status_code == 200:
                        models_data = models_resp.json()
                        self.logger.error(f"   📋 Available models: {models_data}")
                except:
                    pass
                
                # Check if this is GPT-2
                is_gpt2 = "gpt2" in actual_model_name.lower() or "gpt2" in model_name.lower()
                
                # Return detailed error
                error_detail = f"vLLM inference failed: Model '{actual_model_name}' does not support Completions or Chat Completions API"
                self.logger.error(f"   ❌ {error_detail}")
                
                if is_gpt2:
                    self.logger.error(f"   💡 GPT-2 is NOT supported by vLLM OpenAI API endpoints")
                    self.logger.error(f"   💡 SOLUTION: Restart Modelium - GPT-2 will be automatically routed to Ray Serve")
                    self.logger.error(f"   💡 Make sure Ray Serve is enabled in modelium.yaml")
                else:
                    self.logger.error(f"   💡 Suggestion: Model may not be fully supported in vLLM 0.11+")
                    self.logger.error(f"   💡 Alternative: Try using Ray Serve for this model")
                self.logger.error(f"   💡 Check vLLM logs at /tmp/modelium_vllm_logs/ for details")
                
                return {
                    "error": error_detail,
                    "model": actual_model_name,
                    "tried_names": model_names_to_try,
                    "endpoint": endpoint,
                    "suggestion": "GPT-2 is not supported by vLLM. Restart Modelium - it will automatically route GPT-2 to Ray Serve." if is_gpt2 else "Model may not be fully supported in vLLM. Consider using Ray Serve.",
                    "vllm_logs": "/tmp/modelium_vllm_logs/",
                    "is_gpt2": is_gpt2
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
                # Ray Serve endpoint
                endpoint = info.get('endpoint')
                if not endpoint:
                    self.logger.error(f"   ❌ No endpoint found for Ray model")
                    return {"error": "No endpoint configured for Ray model"}
                
                try:
                    resp = requests.post(
                        endpoint,
                        json={
                            "prompt": prompt,
                            "max_tokens": max_tokens,
                            "temperature": kwargs.get("temperature", 0.7)
                        },
                        timeout=60  # Longer timeout for model loading/generation
                    )
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        # Check for errors in response
                        if "error" in result:
                            self.logger.error(f"   ❌ Ray Serve returned error: {result.get('error')}")
                        return result
                    else:
                        self.logger.error(f"   ❌ Ray Serve returned {resp.status_code}: {resp.text}")
                        return {"error": f"Ray Serve returned {resp.status_code}: {resp.text}"}
                except requests.exceptions.Timeout:
                    self.logger.error(f"   ❌ Ray Serve request timed out")
                    return {"error": "Ray Serve request timed out (model may still be loading)"}
                except Exception as e:
                    self.logger.error(f"   ❌ Ray Serve request failed: {e}")
                    return {"error": str(e)}
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            self.logger.error(f"   ❌ Inference failed with exception: {e}")
            self.logger.error(f"   Exception type: {type(e)}")
            self.logger.error(f"   Exception args: {e.args}")
            self.logger.error(f"   Full traceback:\n{error_traceback}")
            return {
                "error": str(e), 
                "error_type": type(e).__name__, 
                "traceback": error_traceback
            }
        
        # Safety: ensure we always return a dict
        self.logger.error(f"   ❌ Inference method reached end without returning (should never happen)")
        return {"error": "Inference method did not return a result"}

