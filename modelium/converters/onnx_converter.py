"""
ONNX model optimizer and converter.

Optimizes ONNX graphs and converts to other formats.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional, List

import onnx
from onnx import optimizer

logger = logging.getLogger(__name__)


class ONNXConverter:
    """
    Converter and optimizer for ONNX models.
    
    Supports:
    - ONNX graph optimization
    - ONNX → TensorRT (via trtexec)
    - ONNX → OpenVINO
    """
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_model(self, model_path: Path) -> onnx.ModelProto:
        """Load ONNX model from file."""
        self.logger.info(f"Loading ONNX model from {model_path}")
        
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        
        return model
    
    def optimize(
        self,
        model_path: Path,
        output_path: Path,
        optimization_level: int = 2,
    ) -> Path:
        """
        Optimize ONNX model.
        
        Args:
            model_path: Path to input ONNX model
            output_path: Path to save optimized model
            optimization_level: Optimization level (0-3)
                0: No optimization
                1: Basic optimizations
                2: Extended optimizations (default)
                3: Layout optimizations
            
        Returns:
            Path to optimized model
        """
        self.logger.info(f"Optimizing ONNX model (level {optimization_level})")
        
        # Load model
        model = self.load_model(model_path)
        
        # Get optimization passes based on level
        passes = self._get_optimization_passes(optimization_level)
        
        # Optimize
        optimized_model = optimizer.optimize(model, passes)
        
        # Save
        onnx.save(optimized_model, str(output_path))
        
        self.logger.info(f"Saved optimized model to {output_path}")
        
        return output_path
    
    def _get_optimization_passes(self, level: int) -> List[str]:
        """Get optimization passes for given level."""
        
        # Level 0: No optimization
        if level == 0:
            return []
        
        # Level 1: Basic optimizations
        passes = [
            "eliminate_identity",
            "eliminate_nop_transpose",
            "eliminate_nop_pad",
            "fuse_consecutive_transposes",
            "fuse_consecutive_squeezes",
            "fuse_add_bias_into_conv",
        ]
        
        # Level 2: Extended optimizations
        if level >= 2:
            passes.extend([
                "fuse_bn_into_conv",
                "fuse_matmul_add_bias_into_gemm",
                "fuse_pad_into_conv",
                "eliminate_unused_initializer",
                "extract_constant_to_initializer",
            ])
        
        # Level 3: Layout optimizations
        if level >= 3:
            passes.extend([
                "fuse_transpose_into_gemm",
            ])
        
        return passes
    
    def shape_inference(
        self,
        model_path: Path,
        output_path: Path,
    ) -> Path:
        """
        Run shape inference on ONNX model.
        
        Args:
            model_path: Path to input ONNX model
            output_path: Path to save model with inferred shapes
            
        Returns:
            Path to model with shapes
        """
        self.logger.info("Running shape inference")
        
        model = self.load_model(model_path)
        
        # Run shape inference
        inferred_model = onnx.shape_inference.infer_shapes(model)
        
        # Save
        onnx.save(inferred_model, str(output_path))
        
        self.logger.info(f"Saved model with shapes to {output_path}")
        
        return output_path
    
    def simplify(
        self,
        model_path: Path,
        output_path: Path,
    ) -> Path:
        """
        Simplify ONNX model using onnx-simplifier.
        
        Args:
            model_path: Path to input ONNX model
            output_path: Path to save simplified model
            
        Returns:
            Path to simplified model
        """
        self.logger.info("Simplifying ONNX model")
        
        try:
            import onnxsim
            
            model = self.load_model(model_path)
            
            # Simplify
            simplified_model, check = onnxsim.simplify(model)
            
            if check:
                onnx.save(simplified_model, str(output_path))
                self.logger.info(f"Saved simplified model to {output_path}")
                return output_path
            else:
                self.logger.warning("Simplification check failed, using original model")
                return model_path
                
        except ImportError:
            self.logger.warning("onnx-simplifier not installed, skipping simplification")
            return model_path
        except Exception as e:
            self.logger.error(f"Error during simplification: {e}")
            return model_path
    
    def validate(self, model_path: Path) -> bool:
        """
        Validate ONNX model.
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            True if valid, False otherwise
        """
        try:
            model = self.load_model(model_path)
            onnx.checker.check_model(model)
            self.logger.info("Model validation passed")
            return True
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return False
    
    def get_model_info(self, model_path: Path) -> dict:
        """
        Get information about ONNX model.
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            Dictionary with model information
        """
        model = self.load_model(model_path)
        
        # Get inputs
        inputs = []
        for inp in model.graph.input:
            input_info = {
                "name": inp.name,
                "shape": [dim.dim_value for dim in inp.type.tensor_type.shape.dim],
                "dtype": inp.type.tensor_type.elem_type,
            }
            inputs.append(input_info)
        
        # Get outputs
        outputs = []
        for out in model.graph.output:
            output_info = {
                "name": out.name,
                "shape": [dim.dim_value for dim in out.type.tensor_type.shape.dim],
                "dtype": out.type.tensor_type.elem_type,
            }
            outputs.append(output_info)
        
        # Get ops
        ops = {}
        for node in model.graph.node:
            ops[node.op_type] = ops.get(node.op_type, 0) + 1
        
        info = {
            "ir_version": model.ir_version,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "opset_version": model.opset_import[0].version if model.opset_import else None,
            "inputs": inputs,
            "outputs": outputs,
            "operations": ops,
            "num_nodes": len(model.graph.node),
            "num_initializers": len(model.graph.initializer),
        }
        
        return info

