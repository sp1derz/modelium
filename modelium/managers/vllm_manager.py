"""
vLLM Runtime Manager

Manages vLLM instances by spawning/killing processes.
Each model gets its own vLLM process on a specific GPU and port.
"""

import logging
import subprocess
import time
import requests
import signal
import psutil
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VLLMInstance:
    """Running vLLM instance."""
    model_name: str
    process: subprocess.Popen
    port: int
    gpu_id: int
    model_path: str
    started_at: float
    pid: int


class VLLMManager:
    """
    Manages vLLM instances via process spawning.
    
    Since vLLM doesn't have a management API for dynamic loading,
    we spawn a separate vLLM process for each model.
    
    Usage:
        manager = VLLMManager(base_port=8100)
        
        # Load a model (spawns vLLM process)
        manager.load_model("gpt2", "/models/repository/gpt2", gpu_id=0)
        
        # Unload (kills process)
        manager.unload_model("gpt2")
    """
    
    def __init__(
        self,
        base_port: int = 8100,
        host: str = "0.0.0.0",
        default_settings: Optional[Dict] = None
    ):
        """
        Initialize vLLM manager.
        
        Args:
            base_port: Starting port for vLLM instances (8100, 8101, 8102...)
            host: Host to bind to
            default_settings: Default vLLM settings (dtype, max_model_len, etc.)
        """
        self.base_port = base_port
        self.host = host
        self.default_settings = default_settings or {
            "dtype": "auto",
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.9,
        }
        
        self._instances: Dict[str, VLLMInstance] = {}
        self._next_port = base_port
        self._used_ports = set()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(f"vLLM Manager initialized (base_port={base_port})")
    
    def load_model(
        self,
        model_name: str,
        model_path: Path,
        gpu_id: int = 0,
        settings: Optional[Dict] = None
    ) -> bool:
        """
        Load a model by spawning a vLLM process.
        
        Args:
            model_name: Name for the model
            model_path: Path to model files (HuggingFace format)
            gpu_id: GPU to use
            settings: vLLM settings override
        
        Returns:
            True if successful
        """
        try:
            if model_name in self._instances:
                self.logger.warning(f"{model_name} already loaded")
                return True
            
            self.logger.info(f"🚀 Starting vLLM for {model_name}...")
            
            # Allocate port
            port = self._allocate_port()
            
            # Merge settings
            final_settings = {**self.default_settings, **(settings or {})}
            
            # Build vLLM command
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(model_path),
                "--host", self.host,
                "--port", str(port),
                "--dtype", final_settings.get("dtype", "auto"),
                "--gpu-memory-utilization", str(final_settings.get("gpu_memory_utilization", 0.9)),
            ]
            
            if final_settings.get("max_model_len"):
                cmd.extend(["--max-model-len", str(final_settings["max_model_len"])])
            
            if final_settings.get("tensor_parallel_size", 1) > 1:
                cmd.extend(["--tensor-parallel-size", str(final_settings["tensor_parallel_size"])])
            
            # Set environment (GPU selection)
            env = {
                **subprocess.os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu_id)
            }
            
            self.logger.info(f"   Command: {' '.join(cmd)}")
            self.logger.info(f"   GPU: {gpu_id}")
            self.logger.info(f"   Port: {port}")
            
            # Spawn process
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # Detach from parent
            )
            
            # Create instance record
            instance = VLLMInstance(
                model_name=model_name,
                process=process,
                port=port,
                gpu_id=gpu_id,
                model_path=str(model_path),
                started_at=time.time(),
                pid=process.pid
            )
            
            self._instances[model_name] = instance
            self._used_ports.add(port)
            
            # Wait for vLLM to be ready
            self.logger.info(f"   Waiting for vLLM to start...")
            if self._wait_for_ready(port, timeout=300):
                self.logger.info(f"   ✅ {model_name} loaded on port {port} (PID: {process.pid})")
                return True
            else:
                self.logger.error(f"   ❌ {model_name} failed to start")
                self.unload_model(model_name)
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading {model_name} with vLLM: {e}")
            return False
    
    def unload_model(self, model_name: str, graceful: bool = True) -> bool:
        """
        Unload a model by killing its vLLM process.
        
        Args:
            model_name: Name of model to unload
            graceful: If True, SIGTERM (graceful). If False, SIGKILL (force)
        
        Returns:
            True if successful
        """
        try:
            if model_name not in self._instances:
                self.logger.warning(f"{model_name} not loaded")
                return False
            
            instance = self._instances[model_name]
            
            self.logger.info(f"🛑 Stopping vLLM for {model_name} (PID: {instance.pid})...")
            
            # Try graceful shutdown first
            if graceful:
                try:
                    # Send SIGTERM
                    process = psutil.Process(instance.pid)
                    process.terminate()
                    
                    # Wait up to 10 seconds for graceful shutdown
                    try:
                        instance.process.wait(timeout=10)
                        self.logger.info(f"   ✅ {model_name} stopped gracefully")
                    except subprocess.TimeoutExpired:
                        self.logger.warning(f"   Graceful shutdown timeout, forcing...")
                        process.kill()
                        instance.process.wait(timeout=5)
                        self.logger.info(f"   ✅ {model_name} force killed")
                        
                except psutil.NoSuchProcess:
                    self.logger.warning(f"   Process {instance.pid} already dead")
            else:
                # Force kill
                instance.process.kill()
                instance.process.wait(timeout=5)
                self.logger.info(f"   ✅ {model_name} force killed")
            
            # Clean up
            self._used_ports.discard(instance.port)
            del self._instances[model_name]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error unloading {model_name}: {e}")
            return False
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded."""
        if model_name not in self._instances:
            return False
        
        instance = self._instances[model_name]
        
        # Check if process is alive
        try:
            process = psutil.Process(instance.pid)
            if not process.is_running():
                # Process died, clean up
                self.logger.warning(f"{model_name} process died unexpectedly")
                self._used_ports.discard(instance.port)
                del self._instances[model_name]
                return False
            return True
        except psutil.NoSuchProcess:
            # Process doesn't exist
            self._used_ports.discard(instance.port)
            del self._instances[model_name]
            return False
    
    def list_loaded_models(self) -> List[str]:
        """List all currently loaded models."""
        # Filter out dead processes
        alive = []
        for model_name in list(self._instances.keys()):
            if self.is_model_loaded(model_name):
                alive.append(model_name)
        return alive
    
    def get_model_endpoint(self, model_name: str) -> Optional[str]:
        """Get the endpoint URL for a loaded model."""
        if model_name in self._instances:
            instance = self._instances[model_name]
            return f"http://{self.host}:{instance.port}"
        return None
    
    def get_instance_stats(self, model_name: str) -> Optional[Dict]:
        """Get stats for a vLLM instance."""
        if model_name not in self._instances:
            return None
        
        instance = self._instances[model_name]
        
        try:
            process = psutil.Process(instance.pid)
            return {
                "model": model_name,
                "pid": instance.pid,
                "port": instance.port,
                "gpu_id": instance.gpu_id,
                "uptime_seconds": time.time() - instance.started_at,
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "status": process.status(),
            }
        except psutil.NoSuchProcess:
            return None
    
    def cleanup_dead_processes(self):
        """Clean up any dead processes."""
        for model_name in list(self._instances.keys()):
            if not self.is_model_loaded(model_name):
                self.logger.warning(f"Cleaning up dead instance: {model_name}")
    
    def shutdown_all(self):
        """Shutdown all vLLM instances."""
        self.logger.info("Shutting down all vLLM instances...")
        for model_name in list(self._instances.keys()):
            self.unload_model(model_name, graceful=True)
        self.logger.info("All vLLM instances stopped")
    
    def _allocate_port(self) -> int:
        """Allocate next available port."""
        port = self._next_port
        while port in self._used_ports:
            port += 1
        self._next_port = port + 1
        return port
    
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

