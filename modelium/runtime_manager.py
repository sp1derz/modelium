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
                    self.logger.warning(f"   vLLM may fail to compile CUDA code without headers")
                    self.logger.warning(f"   Install: sudo yum install python{py_version.split('.')[0]}-devel  # Amazon Linux")
                    self.logger.warning(f"   Or: sudo apt-get install python{py_version.split('.')[0]}-dev  # Ubuntu/Debian")
                    # Don't return False - let vLLM try anyway, it might work
            except Exception as e:
                self.logger.warning(f"   ⚠️  Could not check for Python headers: {e}")
                # Continue anyway - might work
            
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
            
            self.logger.info(f"   Spawning vLLM on port {port}, GPU {gpu_id}")
            
            # Spawn process
            self.logger.info(f"   Command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                text=True
            )
            
            # Wait for ready
            self.logger.info(f"   Waiting for vLLM to start (PID: {process.pid})...")
            if self._wait_for_vllm_ready(port, timeout=180, process=process):
                self._vllm_processes[model_name] = process
                self._loaded_models[model_name] = {
                    "runtime": "vllm",
                    "endpoint": f"http://localhost:{port}",
                    "port": port,
                    "gpu": gpu_id,
                    "pid": process.pid,
                }
                self.logger.info(f"   ✅ {model_name} ready on port {port}")
                return True
            else:
                self.logger.error(f"   ❌ vLLM failed to start on port {port}")
                # Check if process is still running
                if process.poll() is None:
                    self.logger.error(f"   Process still running but not responding")
                else:
                    self.logger.error(f"   Process exited with code: {process.returncode}")
                    # Read stderr for error details (read all available)
                    try:
                        # Try to read stderr (non-blocking)
                        import select
                        if select.select([process.stderr], [], [], 0)[0]:
                            stderr_output = process.stderr.read()
                            if stderr_output:
                                # Log first 2000 chars of stderr
                                self.logger.error(f"   vLLM stderr (first 2000 chars):")
                                for line in stderr_output[:2000].split('\n')[:50]:  # First 50 lines
                                    if line.strip():
                                        self.logger.error(f"      {line}")
                    except Exception as e:
                        self.logger.error(f"   Could not read stderr: {e}")
                        # Try alternative: communicate with timeout
                        try:
                            _, stderr = process.communicate(timeout=1)
                            if stderr:
                                self.logger.error(f"   vLLM stderr: {stderr[:1000]}")
                        except:
                            pass
                try:
                    process.kill()
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
    
    def _wait_for_vllm_ready(self, port: int, timeout: int, process: Optional[subprocess.Popen] = None) -> bool:
        """Wait for vLLM to be ready."""
        start = time.time()
        check_count = 0
        while time.time() - start < timeout:
            # Check if process died
            if process and process.poll() is not None:
                self.logger.error(f"   vLLM process died with code {process.returncode}")
                try:
                    stderr_output = process.stderr.read()
                    if stderr_output:
                        self.logger.error(f"   vLLM stderr: {stderr_output[:1000]}")
                except:
                    pass
                return False
            
            # Check health endpoint
            try:
                resp = requests.get(f"http://localhost:{port}/health", timeout=2)
                if resp.status_code == 200:
                    elapsed = time.time() - start
                    self.logger.info(f"   ✅ vLLM ready after {elapsed:.1f}s")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            check_count += 1
            if check_count % 10 == 0:  # Log every 20 seconds
                elapsed = time.time() - start
                self.logger.info(f"   Still waiting... ({elapsed:.0f}s elapsed)")
            
            time.sleep(2)
        
        self.logger.error(f"   Timeout after {timeout}s")
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
                # vLLM uses OpenAI format
                resp = requests.post(
                    f"{info['endpoint']}/v1/completions",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        **kwargs
                    },
                    timeout=30
                )
                return resp.json()
            
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

