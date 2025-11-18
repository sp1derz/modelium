"""
Inference server for the Modelium LLM.

Provides a FastAPI endpoint for generating conversion plans.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from modelium.core.descriptor import ModelDescriptor
from modelium.modelium_llm.schemas import ConversionPlan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Modelium LLM Inference Server")


class GenerateRequest(BaseModel):
    """Request for generating a conversion plan."""
    
    model_descriptor: Dict[str, Any]
    target_environment: str = "kubernetes"
    gpu_type: str = "nvidia-a100"
    max_latency_ms: int = 100
    expected_qps: int = 100
    batch_size: str = "dynamic"
    precision: str = "fp16"
    additional_context: str = ""
    max_tokens: int = 4096
    temperature: float = 0.3


class GenerateResponse(BaseModel):
    """Response containing the generated conversion plan."""
    
    conversion_plan: Dict[str, Any]
    raw_output: str
    generation_time_seconds: float


class ModeliumLLMInference:
    """Inference engine for the Modelium LLM."""
    
    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        
        logger.info(f"Loading model from {model_path}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        self.model.eval()
        
        # Load prompts
        prompts_dir = Path("modelium/modelium_llm/prompts")
        with open(prompts_dir / "system_prompt.txt", "r") as f:
            self.system_prompt = f.read()
        
        with open(prompts_dir / "user_prompt_template.txt", "r") as f:
            self.user_template = f.read()
        
        logger.info("Model loaded successfully")
    
    def generate_plan(
        self,
        model_descriptor: Dict[str, Any],
        target_environment: str = "kubernetes",
        gpu_type: str = "nvidia-a100",
        max_latency_ms: int = 100,
        expected_qps: int = 100,
        batch_size: str = "dynamic",
        precision: str = "fp16",
        additional_context: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """Generate a conversion plan."""
        
        import time
        start_time = time.time()
        
        # Format user prompt
        user_prompt = self.user_template.format(
            model_descriptor=json.dumps(model_descriptor, indent=2),
            target_environment=target_environment,
            gpu_type=gpu_type,
            max_latency_ms=max_latency_ms,
            expected_qps=expected_qps,
            batch_size=batch_size,
            precision=precision,
            additional_context=additional_context,
        )
        
        # Create conversation
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Format for model
        prompt = self._format_conversation(conversation)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
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
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        # Parse JSON from output
        try:
            # Try to extract JSON from the response
            plan_dict = self._extract_json(generated_text)
        except Exception as e:
            logger.error(f"Error parsing generated plan: {e}")
            logger.error(f"Generated text: {generated_text}")
            raise ValueError(f"Failed to parse conversion plan: {e}")
        
        generation_time = time.time() - start_time
        
        return {
            "conversion_plan": plan_dict,
            "raw_output": generated_text,
            "generation_time_seconds": generation_time,
        }
    
    def _format_conversation(self, conversation: list[dict]) -> str:
        """Format conversation for model."""
        formatted = ""
        
        for message in conversation:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        # Add assistant start for generation
        formatted += "<|im_start|>assistant\n"
        
        return formatted
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from generated text."""
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
            # Try to find JSON directly
            json_str = text.strip()
        
        # Parse JSON
        return json.loads(json_str)


# Global inference engine
inference_engine: Optional[ModeliumLLMInference] = None


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the inference engine on startup."""
    global inference_engine
    
    import os
    model_path = os.getenv("MODEL_PATH", "./models/modelium-llm-finetuned")
    inference_engine = ModeliumLLMInference(model_path)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate a conversion plan from a model descriptor."""
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = inference_engine.generate_plan(
            model_descriptor=request.model_descriptor,
            target_environment=request.target_environment,
            gpu_type=request.gpu_type,
            max_latency_ms=request.max_latency_ms,
            expected_qps=request.expected_qps,
            batch_size=request.batch_size,
            precision=request.precision,
            additional_context=request.additional_context,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        
        return GenerateResponse(**result)
        
    except Exception as e:
        logger.error(f"Error generating plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": inference_engine is not None}


if __name__ == "__main__":
    import os
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

