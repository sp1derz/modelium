"""
vLLM Service Wrapper

Manages vLLM model loading and inference.
"""

import logging
import subprocess
import time
import requests
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class VLLMService:
    """
    Wrapper for vLLM inference engine.
    
    Handles model loading, process management, and inference requests.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Initialize vLLM service.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        self.host = host
        self.port = port
        self._processes: Dict[str, subprocess.Popen] = {}
        self._model_ports: Dict[str, int] = {}
        self._next_port = port
    
    def load_model(
        self,
        model_name: str,
        model_path: str,
        gpu_id: int = 0,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load a model with vLLM.
        
        Args:
            model_name: Name to identify the model
            model_path: Path to model file or HuggingFace repo
            gpu_id: GPU to load on
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Data type (auto, float16, bfloat16)
            max_model_len: Maximum sequence length
        
        Returns:
            Dict with status and endpoint info
        """
        if model_name in self._processes:
            logger.warning(f"Model {model_name} already loaded")
            return {
                "status": "already_loaded",
                "endpoint": f"http://{self.host}:{self._model_ports[model_name]}",
            }
        
        # Allocate port
        model_port = self._next_port
        self._next_port += 1
        
        try:
            logger.info(f"⚡ Loading {model_name} with vLLM on GPU {gpu_id}...")
            
            # Build vLLM command
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_path,
                "--host", self.host,
                "--port", str(model_port),
                "--tensor-parallel-size", str(tensor_parallel_size),
                "--dtype", dtype,
            ]
            
            if max_model_len:
                cmd.extend(["--max-model-len", str(max_model_len)])
            
            # Set GPU
            env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
            
            # Start process
            process = subprocess.Popen(
                cmd,
                env={**subprocess.os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            self._processes[model_name] = process
            self._model_ports[model_name] = model_port
            
            # Wait for model to be ready
            logger.info(f"   Waiting for vLLM to be ready...")
            if self._wait_for_ready(model_port, timeout=300):
                logger.info(f"   ✅ {model_name} ready at port {model_port}")
                return {
                    "status": "loaded",
                    "endpoint": f"http://{self.host}:{model_port}",
                    "port": model_port,
                }
            else:
                logger.error(f"   ❌ {model_name} failed to start")
                self.unload_model(model_name)
                return {"status": "error", "error": "Failed to start"}
                
        except Exception as e:
            logger.error(f"   ❌ Error loading {model_name}: {e}")
            return {"status": "error", "error": str(e)}
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model.
        
        Args:
            model_name: Name of model to unload
        
        Returns:
            True if successful
        """
        if model_name not in self._processes:
            logger.warning(f"Model {model_name} not loaded")
            return False
        
        try:
            logger.info(f"🔻 Unloading {model_name}...")
            process = self._processes[model_name]
            process.terminate()
            process.wait(timeout=10)
            
            del self._processes[model_name]
            del self._model_ports[model_name]
            
            logger.info(f"   ✅ {model_name} unloaded")
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Error unloading {model_name}: {e}")
            return False
    
    def inference(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Run inference on a loaded model.
        
        Args:
            model_name: Name of loaded model
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Dict with generated text
        """
        if model_name not in self._model_ports:
            return {"error": "Model not loaded"}
        
        port = self._model_ports[model_name]
        url = f"http://{self.host}:{port}/v1/completions"
        
        try:
            response = requests.post(
                url,
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Inference error for {model_name}: {e}")
            return {"error": str(e)}
    
    def get_loaded_models(self) -> list:
        """Get list of loaded models."""
        return list(self._processes.keys())
    
    def _wait_for_ready(self, port: int, timeout: int = 300) -> bool:
        """Wait for vLLM server to be ready."""
        url = f"http://{self.host}:{port}/health"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(2)
        
        return False

