"""
Unified Modelium Brain - One LLM for all intelligent decisions.

This is THE brain of Modelium. It handles:
- Task 1: Model analysis and conversion plan generation
- Task 2: GPU orchestration decisions (load/unload models)

Model: Fine-tuned Qwen-2.5-1.8B (~2GB, runs on any GPU)
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from modelium.brain.prompts import (
    CONVERSION_SYSTEM_PROMPT,
    ORCHESTRATION_SYSTEM_PROMPT,
    format_conversion_prompt,
    format_orchestration_prompt,
)

logger = logging.getLogger(__name__)


class ModeliumBrain:
    """
    The unified brain of Modelium.
    
    One model, two tasks:
    1. generate_conversion_plan(): Analyze model → deployment strategy
    2. make_orchestration_decision(): Current state → load/unload actions
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda:0",
        dtype: str = "float16",
        fallback_to_rules: bool = True,
    ):
        """
        Initialize the Modelium Brain.
        
        Args:
            model_name: HuggingFace model name or local path
            device: Device to run on (cuda:0, cpu, etc.)
            dtype: Model dtype (float16, float32, bfloat16)
            fallback_to_rules: If True, use rule-based logic when LLM fails
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.fallback_to_rules = fallback_to_rules
        self.model = None
        self.tokenizer = None
        
        logger.info(f"🧠 Initializing Modelium Brain: {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the LLM from HuggingFace or local path."""
        try:
            # Convert dtype string to torch dtype
            torch_dtype = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }.get(self.dtype, torch.float16)
            
            logger.info(f"   Loading model on {self.device} ({self.dtype})...")
            logger.info(f"   Model will be downloaded/cached from HuggingFace: {self.model_name}")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=self.device,
                trust_remote_code=True,
            )
            
            # Show where model was cached
            try:
                from transformers.utils import TRANSFORMERS_CACHE
                import os
                cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                logger.info(f"   ✅ Model cached at: {cache_dir}/hub/models--{self.model_name.replace('/', '--')}")
            except:
                pass
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Get model size
            param_count = sum(p.numel() for p in self.model.parameters())
            size_gb = param_count * 2 / 1e9  # FP16 = 2 bytes per param
            
            logger.info(f"   ✅ Brain loaded: {param_count/1e9:.1f}B params, {size_gb:.1f}GB VRAM")
            
            # Verify model is actually on the requested device
            if self.device.startswith("cuda"):
                first_param = next(self.model.parameters(), None)
                if first_param is not None:
                    actual_device = str(first_param.device)
                    if not actual_device.startswith("cuda"):
                        logger.warning(f"   ⚠️  Brain requested {self.device} but loaded on {actual_device}")
                    else:
                        logger.info(f"   ✅ Brain verified on {actual_device}")
            
        except Exception as e:
            logger.error(f"   ❌ Failed to load brain: {e}")
            if not self.fallback_to_rules:
                raise
            logger.warning("   ⚠️  Will use rule-based fallback for decisions")
    
    def generate_conversion_plan(
        self,
        model_descriptor: Dict[str, Any],
        available_gpus: int = 0,
        gpu_memory: Optional[List[int]] = None,
        target_environment: str = "kubernetes",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a deployment plan for a newly discovered model.
        
        This is Task 1 of the brain: analyze model characteristics and decide:
        - Best runtime (vLLM, Ray Serve, TensorRT)
        - Which GPU to deploy to
        - Configuration settings
        - Estimated resource requirements
        
        Args:
            model_descriptor: Dict with model metadata (from analyzer)
            available_gpus: Number of GPUs available
            gpu_memory: List of available memory per GPU (in GB)
            target_environment: Target deployment environment
            **kwargs: Additional context
        
        Returns:
            Dict with conversion plan:
            {
                "runtime": "vllm",
                "target_gpu": 1,
                "config": {...},
                "reasoning": "...",
                "confidence": 0.92
            }
        """
        start_time = time.time()
        
        logger.info(f"🧠 Brain: Analyzing {model_descriptor.get('name', 'unknown')}...")
        
        # Try LLM first
        if self.model is not None:
            try:
                plan = self._llm_generate_conversion_plan(
                    model_descriptor, available_gpus, gpu_memory, target_environment, **kwargs
                )
                
                elapsed = time.time() - start_time
                logger.info(f"   ✅ Plan generated in {elapsed:.2f}s (confidence: {plan.get('confidence', 0):.2f})")
                
                return plan
                
            except Exception as e:
                logger.error(f"   ❌ LLM conversion planning failed: {e}")
                if not self.fallback_to_rules:
                    raise
        
        # Fallback to rule-based
        logger.info("   ⚠️  Using rule-based conversion planning...")
        return self._rule_based_conversion_plan(
            model_descriptor, available_gpus, gpu_memory
        )
    
    def make_orchestration_decision(
        self,
        current_state: Dict[str, Any],
        policies: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Make orchestration decisions based on current GPU/model state.
        
        This is Task 2 of the brain: analyze current state and decide:
        - Which models to keep loaded
        - Which models to evict
        - Which pending models to load
        - GPU allocation strategy
        
        Args:
            current_state: Dict with:
                - models_loaded: List of loaded models with metrics
                - models_unloaded: List of unloaded models with queue info
                - gpu_memory: GPU memory usage
                - pending_requests: Queued requests per model
            policies: Dict with:
                - evict_after_idle_seconds: Idle threshold
                - always_loaded: Models to never evict
                - priority_by_qps: Prioritize by traffic
        
        Returns:
            Dict with orchestration actions:
            {
                "actions": [
                    {"action": "evict", "model": "...", "from_gpu": 2, "reasoning": "..."},
                    {"action": "load", "model": "...", "to_gpu": 1, "reasoning": "..."},
                ],
                "predicted_metrics": {...},
                "confidence": 0.89
            }
        """
        logger.debug("🧠 Brain: Making orchestration decision...")
        
        # Try LLM first
        if self.model is not None:
            try:
                decision = self._llm_make_orchestration_decision(
                    current_state, policies
                )
                
                logger.debug(f"   ✅ Decision: {len(decision.get('actions', []))} actions")
                return decision
                
            except Exception as e:
                logger.error(f"   ❌ LLM orchestration failed: {e}")
                if not self.fallback_to_rules:
                    raise
        
        # Fallback to rule-based
        logger.debug("   ⚠️  Using rule-based orchestration...")
        return self._rule_based_orchestration(current_state, policies)
    
    def _llm_generate_conversion_plan(
        self,
        model_descriptor: Dict[str, Any],
        available_gpus: int,
        gpu_memory: Optional[List[int]],
        target_environment: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Use LLM to generate conversion plan."""
        # Format the prompt
        user_prompt = format_conversion_prompt(
            model_descriptor=model_descriptor,
            available_gpus=available_gpus,
            gpu_memory=gpu_memory or [],
            target_environment=target_environment,
            **kwargs,
        )
        
        # Generate with LLM
        output = self._generate(
            system_prompt=CONVERSION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=2048,
            temperature=0.3,
        )
        
        # Parse JSON from output
        plan = self._extract_json(output)
        
        return plan
    
    def _llm_make_orchestration_decision(
        self,
        current_state: Dict[str, Any],
        policies: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use LLM to make orchestration decision."""
        # Format the prompt
        user_prompt = format_orchestration_prompt(
            current_state=current_state,
            policies=policies,
        )
        
        # Log the full prompt being sent to brain
        logger.info("=" * 80)
        logger.info("🧠 BRAIN PROMPT (Full):")
        logger.info("=" * 80)
        logger.info("SYSTEM PROMPT:")
        logger.info(ORCHESTRATION_SYSTEM_PROMPT)
        logger.info("=" * 80)
        logger.info("USER PROMPT (Current State + Policies):")
        logger.info(user_prompt)
        logger.info("=" * 80)
        
        # Generate with LLM
        output = self._generate(
            system_prompt=ORCHESTRATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=2048,
            temperature=0.3,
        )
        
        # Log brain's response
        logger.info("🧠 BRAIN RESPONSE (Raw):")
        logger.info(output)
        logger.info("=" * 80)
        
        # Parse JSON from output
        decision = self._extract_json(output)
        
        # Normalize decision format: handle both list and dict responses
        # Brain might return: [{"action": "keep", ...}] or {"actions": [{"action": "keep", ...}]}
        if isinstance(decision, list):
            # Brain returned a list directly, wrap it in dict
            logger.warning("🧠 Brain returned list instead of dict - normalizing format")
            decision = {"actions": decision}
        elif isinstance(decision, dict) and "actions" not in decision:
            # Brain returned dict but without "actions" key
            logger.warning("🧠 Brain returned dict without 'actions' key - normalizing format")
            # Try to find actions in the dict
            if "action" in decision or any(isinstance(v, list) for v in decision.values()):
                # Might be a single action or actions in a different key
                decision = {"actions": [decision] if "action" in decision else list(decision.values())[0]}
            else:
                # Fallback: wrap in actions
                decision = {"actions": []}
        
        # Log parsed decision
        import json
        logger.info("🧠 BRAIN DECISION (Parsed):")
        logger.info(json.dumps(decision, indent=2))
        logger.info("=" * 80)
        
        return decision
    
    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """Generate text from LLM."""
        # Format conversation
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Format for Qwen chat template
        formatted = self._format_conversation(conversation)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return generated
    
    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation for Qwen chat template."""
        formatted = ""
        
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        # Start assistant response
        formatted += "<|im_start|>assistant\n"
        
        return formatted
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM output."""
        # Try to find JSON in code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            json_str = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            json_str = text[start:end].strip()
        else:
            json_str = text.strip()
        
        # Parse JSON
        return json.loads(json_str)
    
    # ========================================================================
    # Rule-based fallbacks (when LLM not available or fails)
    # ========================================================================
    
    def _rule_based_conversion_plan(
        self,
        model_descriptor: Dict[str, Any],
        available_gpus: int,
        gpu_memory: Optional[List[int]],
    ) -> Dict[str, Any]:
        """Simple rule-based conversion planning (fallback)."""
        model_type = model_descriptor.get("model_type", "unknown").lower()
        model_size_gb = model_descriptor.get("resources", {}).get("memory_bytes", 0) / 1e9
        framework = model_descriptor.get("framework", "unknown").lower()
        
        # Detect if it's an LLM
        is_llm = (
            model_type in ["language_model", "llm", "causal_lm"] or
            "llm" in model_descriptor.get("name", "").lower() or
            model_size_gb > 5
        )
        
        # Choose runtime
        if is_llm and available_gpus > 0:
            runtime = "vllm"
        elif framework == "pytorch" and available_gpus > 0 and model_size_gb < 2:
            runtime = "tensorrt"
        elif available_gpus > 0:
            runtime = "ray_serve"
        else:
            runtime = "ray_serve"  # CPU fallback
        
        # Choose GPU (simple: GPU with most free memory)
        target_gpu = 0
        if gpu_memory and len(gpu_memory) > 0:
            target_gpu = gpu_memory.index(max(gpu_memory))
        
        return {
            "runtime": runtime,
            "target_gpu": target_gpu,
            "config": {
                "dtype": "float16" if available_gpus > 0 else "float32",
            },
            "reasoning": f"Rule-based: Detected {model_type}, chose {runtime}",
            "confidence": 0.7,  # Lower confidence for rules
            "estimated_load_time_seconds": 60,
        }
    
    def _rule_based_orchestration(
        self,
        current_state: Dict[str, Any],
        policies: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simple rule-based orchestration (fallback)."""
        actions = []
        
        evict_threshold = policies.get("evict_after_idle_seconds", 300)
        always_loaded = policies.get("always_loaded", [])
        
        # Rule 1: Evict idle models
        for model in current_state.get("models_loaded", []):
            name = model["name"]
            idle_seconds = model.get("idle_seconds", 0)
            qps = model.get("qps", 0)
            
            if name not in always_loaded and idle_seconds > evict_threshold and qps == 0:
                actions.append({
                    "action": "evict",
                    "model": name,
                    "from_gpu": model["gpu"],
                    "reasoning": f"Idle {idle_seconds}s (>{evict_threshold}s), 0 QPS"
                })
        
        # Rule 2: Load models with pending requests
        for model in current_state.get("models_unloaded", []):
            name = model["name"]
            pending = model.get("pending_requests", 0)
            
            if pending > 0:
                # Find GPU with most free memory
                gpu_memory = current_state.get("gpu_memory", {})
                target_gpu = 0
                max_free = 0
                for gpu_id, mem_info in gpu_memory.items():
                    free = mem_info["total"] - mem_info["used"]
                    if free > max_free:
                        max_free = free
                        target_gpu = int(gpu_id.split("_")[1])  # "gpu_0" -> 0
                
                actions.append({
                    "action": "load",
                    "model": name,
                    "to_gpu": target_gpu,
                    "reasoning": f"{pending} pending requests, loading to GPU {target_gpu}"
                })
        
        return {
            "actions": actions,
            "predicted_metrics": {},
            "confidence": 0.6,  # Lower confidence for rules
        }

