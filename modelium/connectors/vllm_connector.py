"""
vLLM HTTP Connector

Connects to an external vLLM server (OpenAI-compatible API).
"""

import logging
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class VLLMConnector:
    """
    HTTP client for vLLM inference server.
    
    Connects to vLLM's OpenAI-compatible API.
    Users should start vLLM separately:
    
        docker run --gpus all -p 8001:8000 \\
            vllm/vllm-openai:latest \\
            --model gpt2 \\
            --dtype auto
    
    Then Modelium connects to it for inference.
    """
    
    def __init__(self, endpoint: str, timeout: int = 300):
        """
        Initialize vLLM connector.
        
        Args:
            endpoint: Base URL of vLLM server (e.g., "http://localhost:8001")
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def health_check(self) -> bool:
        """
        Check if vLLM server is healthy.
        
        Returns:
            True if server is healthy and ready
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"vLLM health check failed: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """
        List models loaded in vLLM.
        
        Returns:
            List of model names
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/v1/models",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except Exception as e:
            self.logger.error(f"Failed to list vLLM models: {e}")
            return []
    
    def inference(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference using vLLM's OpenAI-compatible API.
        
        Args:
            model: Model name (as loaded in vLLM)
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            stream: Whether to stream responses
            **kwargs: Additional vLLM parameters
        
        Returns:
            Dict with generation results (OpenAI format)
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream,
                **kwargs
            }
            
            response = self.session.post(
                f"{self.endpoint}/v1/completions",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            self.logger.error(f"vLLM inference timeout after {self.timeout}s")
            return {"error": "Inference timeout", "model": model}
        except Exception as e:
            self.logger.error(f"vLLM inference failed: {e}")
            return {"error": str(e), "model": model}
    
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat completion using vLLM's OpenAI-compatible chat API.
        
        Args:
            model: Model name
            messages: List of chat messages [{"role": "user", "content": "..."}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
        
        Returns:
            Dict with chat completion results
        """
        try:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            response = self.session.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"vLLM chat completion failed: {e}")
            return {"error": str(e), "model": model}
    
    def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Args:
            model: Model name
        
        Returns:
            Model info dict or None if not found
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/v1/models/{model}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.debug(f"Failed to get vLLM model info: {e}")
            return None

