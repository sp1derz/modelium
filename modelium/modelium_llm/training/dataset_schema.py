"""
Training dataset schema for Modelium LLM.

Each training example consists of a model descriptor and a corresponding
conversion plan that was successfully executed.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class TrainingExample(BaseModel):
    """
    A single training example for the Modelium LLM.
    
    The model learns to map descriptors to successful conversion plans.
    """
    
    # Input: Model descriptor
    model_descriptor: Dict[str, Any]
    
    # Deployment requirements
    target_environment: str = "kubernetes"
    gpu_type: str = "nvidia-a100"
    max_latency_ms: int = 100
    expected_qps: int = 100
    batch_size: str = "dynamic"
    precision: str = "fp16"
    
    # Additional context
    additional_context: str = ""
    
    # Output: Conversion plan
    conversion_plan: Dict[str, Any]
    
    # Metadata
    execution_status: str = "success"  # success, failure, partial
    actual_conversion_time_minutes: Optional[int] = None
    actual_accuracy_impact: Optional[float] = None
    
    # Tags for filtering
    tags: list[str] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_descriptor": {
                    "id": "resnet50",
                    "framework": "pytorch",
                    "model_type": "vision",
                },
                "conversion_plan": {
                    "target_format": "tensorrt_fp16",
                    "steps": [],
                },
                "execution_status": "success",
                "tags": ["vision", "pytorch", "tensorrt"],
            }
        }


def create_training_examples() -> list[TrainingExample]:
    """
    Create synthetic training examples.
    
    In production, these would come from successful conversions in the system.
    """
    
    examples = []
    
    # Example 1: PyTorch ResNet50 → TensorRT FP16
    examples.append(TrainingExample(
        model_descriptor={
            "id": "resnet50-imagenet",
            "name": "ResNet50",
            "framework": "pytorch",
            "model_type": "vision",
            "architecture": "ResNet",
            "input_shapes": {"input": [1, 3, 224, 224]},
            "output_shapes": {"output": [1, 1000]},
            "resources": {
                "parameters": 25557032,
                "memory_bytes": 102228128,
            },
        },
        gpu_type="nvidia-t4",
        max_latency_ms=50,
        conversion_plan={
            "plan_id": "plan-001",
            "model_id": "resnet50-imagenet",
            "target_format": "tensorrt_fp16",
            "optimization_strategy": "graph_optimization",
            "batching_strategy": "dynamic",
            "steps": [
                {
                    "name": "export_onnx",
                    "description": "Export PyTorch to ONNX",
                    "script": "torch.onnx.export(...)",
                },
                {
                    "name": "convert_tensorrt",
                    "description": "Build TensorRT engine",
                    "command": "trtexec --onnx=model.onnx --saveEngine=model.plan --fp16",
                },
            ],
            "required_memory_gb": 4.0,
            "required_gpu_memory_gb": 2.0,
        },
        execution_status="success",
        actual_conversion_time_minutes=8,
        tags=["pytorch", "vision", "tensorrt", "resnet"],
    ))
    
    # Example 2: HuggingFace BERT → ONNX → TensorRT
    examples.append(TrainingExample(
        model_descriptor={
            "id": "bert-base-uncased",
            "name": "BERT Base",
            "framework": "huggingface",
            "model_type": "nlp",
            "architecture": "BertForSequenceClassification",
            "input_shapes": {
                "input_ids": [1, 128],
                "attention_mask": [1, 128],
            },
            "output_shapes": {"logits": [1, 2]},
            "tokenizer": {
                "type": "BertTokenizer",
                "vocab_size": 30522,
            },
            "resources": {
                "parameters": 110000000,
                "memory_bytes": 440000000,
            },
        },
        gpu_type="nvidia-a100",
        max_latency_ms=20,
        conversion_plan={
            "plan_id": "plan-002",
            "model_id": "bert-base-uncased",
            "target_format": "tensorrt_fp16",
            "optimization_strategy": "graph_optimization",
            "batching_strategy": "dynamic",
            "steps": [
                {
                    "name": "export_onnx",
                    "description": "Export HuggingFace model to ONNX",
                    "command": "python -m transformers.onnx --model=bert-base-uncased --feature=sequence-classification model_onnx/",
                },
                {
                    "name": "optimize_onnx",
                    "description": "Optimize ONNX for transformers",
                    "command": "python -m onnxruntime.transformers.optimizer --input model.onnx --output model_opt.onnx --model_type bert",
                },
                {
                    "name": "convert_tensorrt",
                    "description": "Build TensorRT engine",
                    "command": "trtexec --onnx=model_opt.onnx --saveEngine=model.plan --fp16 --minShapes=input_ids:1x1,attention_mask:1x1 --optShapes=input_ids:8x128,attention_mask:8x128 --maxShapes=input_ids:32x128,attention_mask:32x128",
                },
            ],
            "required_memory_gb": 8.0,
            "required_gpu_memory_gb": 4.0,
        },
        execution_status="success",
        actual_conversion_time_minutes=15,
        tags=["huggingface", "nlp", "tensorrt", "bert"],
    ))
    
    # Example 3: GPT-2 → TRT-LLM
    examples.append(TrainingExample(
        model_descriptor={
            "id": "gpt2-medium",
            "name": "GPT-2 Medium",
            "framework": "huggingface",
            "model_type": "nlp",
            "architecture": "GPT2LMHeadModel",
            "input_shapes": {"input_ids": [1, None]},
            "output_shapes": {"logits": [1, None, 50257]},
            "tokenizer": {
                "type": "GPT2Tokenizer",
                "vocab_size": 50257,
            },
            "resources": {
                "parameters": 354823168,
                "memory_bytes": 1419292672,
            },
        },
        gpu_type="nvidia-a100",
        max_latency_ms=2000,
        precision="fp16",
        conversion_plan={
            "plan_id": "plan-003",
            "model_id": "gpt2-medium",
            "target_format": "trt_llm",
            "optimization_strategy": "quantization",
            "batching_strategy": "continuous",
            "steps": [
                {
                    "name": "convert_checkpoint",
                    "description": "Convert HF checkpoint to TRT-LLM format",
                    "command": "python convert_checkpoint.py --model_dir gpt2-medium --output_dir trt_checkpoint --dtype float16",
                },
                {
                    "name": "build_engine",
                    "description": "Build TRT-LLM engine",
                    "command": "trtllm-build --checkpoint_dir trt_checkpoint --output_dir trt_engines --gemm_plugin float16 --max_batch_size 8 --max_input_len 1024 --max_output_len 512",
                },
            ],
            "required_memory_gb": 16.0,
            "required_gpu_memory_gb": 8.0,
            "notes": ["Uses PagedAttention for efficient memory management"],
        },
        execution_status="success",
        actual_conversion_time_minutes=25,
        tags=["huggingface", "nlp", "trt-llm", "gpt"],
    ))
    
    # Example 4: ONNX Vision Model → TensorRT INT8
    examples.append(TrainingExample(
        model_descriptor={
            "id": "yolov5s",
            "name": "YOLOv5 Small",
            "framework": "onnx",
            "model_type": "vision",
            "input_shapes": {"images": [1, 3, 640, 640]},
            "output_shapes": {"output": [1, 25200, 85]},
            "operations": [
                {"name": "conv1", "type": "Conv"},
                {"name": "maxpool1", "type": "MaxPool"},
            ],
            "resources": {
                "parameters": 7235389,
                "memory_bytes": 28941556,
            },
        },
        gpu_type="nvidia-jetson-xavier",
        max_latency_ms=30,
        precision="int8",
        conversion_plan={
            "plan_id": "plan-004",
            "model_id": "yolov5s",
            "target_format": "tensorrt_int8",
            "optimization_strategy": "quantization",
            "batching_strategy": "static",
            "steps": [
                {
                    "name": "calibrate",
                    "description": "Generate INT8 calibration cache",
                    "script": "# Python script for calibration with sample images",
                },
                {
                    "name": "convert_tensorrt_int8",
                    "description": "Build TensorRT INT8 engine",
                    "command": "trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.plan --int8 --calib=calibration.cache",
                },
            ],
            "required_memory_gb": 4.0,
            "required_gpu_memory_gb": 2.0,
            "warnings": ["INT8 quantization may impact accuracy, validate carefully"],
        },
        execution_status="success",
        actual_conversion_time_minutes=12,
        actual_accuracy_impact=-0.015,
        tags=["onnx", "vision", "tensorrt", "int8", "yolo"],
    ))
    
    return examples

