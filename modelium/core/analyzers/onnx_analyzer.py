"""
ONNX-specific model analyzer.

Extracts graph structure, operations, and metadata from ONNX models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import onnx
from onnx import numpy_helper

from modelium.core.descriptor import OpInfo, ResourceEstimate, ModelType

logger = logging.getLogger(__name__)


class AnalysisResult:
    """Container for ONNX analysis results."""
    
    def __init__(self) -> None:
        self.model_type: Optional[ModelType] = None
        self.operations: List[OpInfo] = []
        self.input_info: Optional[Dict[str, Any]] = None
        self.output_info: Optional[Dict[str, Any]] = None
        self.resources: Optional[ResourceEstimate] = None
        self.metadata: Dict[str, Any] = {}


class ONNXAnalyzer:
    """
    Analyzer for ONNX models.
    
    Extracts graph structure, operations, and shapes.
    """
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze(self, model_path: Path) -> AnalysisResult:
        """
        Analyze ONNX model.
        
        Args:
            model_path: Path to ONNX model file
            
        Returns:
            AnalysisResult with extracted information
        """
        result = AnalysisResult()
        
        try:
            self.logger.info(f"Loading ONNX model from {model_path}")
            model = onnx.load(str(model_path))
            
            # Verify model
            onnx.checker.check_model(model)
            
            # Extract graph information
            graph = model.graph
            
            # Extract operations
            result.operations = self._extract_operations(graph)
            
            # Extract input/output info
            result.input_info = self._extract_io_info(graph.input)
            result.output_info = self._extract_io_info(graph.output)
            
            # Calculate resources
            result.resources = self._calculate_resources(graph)
            
            # Extract metadata
            result.metadata = self._extract_metadata(model)
            
            # Infer model type
            result.model_type = self._infer_model_type(result.operations)
            
            self.logger.info(f"Found {len(result.operations)} operations")
            
        except Exception as e:
            self.logger.error(f"Error analyzing ONNX model: {e}")
        
        return result
    
    def _extract_operations(self, graph: onnx.GraphProto) -> List[OpInfo]:
        """Extract operation information from graph."""
        operations = []
        
        for node in graph.node:
            # Get input/output shapes (if available from value_info)
            input_shapes = []
            output_shapes = []
            
            # Extract attributes
            attributes = {}
            for attr in node.attribute:
                attributes[attr.name] = self._extract_attribute_value(attr)
            
            # Check if this is a custom op
            is_custom = not self._is_standard_op(node.op_type)
            
            op_info = OpInfo(
                name=node.name or f"{node.op_type}_{len(operations)}",
                type=node.op_type,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                attributes=attributes,
                custom=is_custom,
            )
            
            operations.append(op_info)
        
        return operations
    
    def _extract_io_info(self, value_infos: List[Any]) -> Dict[str, Any]:
        """Extract input/output information."""
        names = []
        shapes = {}
        dtypes = {}
        
        for value_info in value_infos:
            name = value_info.name
            names.append(name)
            
            # Extract shape
            if value_info.type.HasField("tensor_type"):
                tensor_type = value_info.type.tensor_type
                shape = []
                for dim in tensor_type.shape.dim:
                    if dim.HasField("dim_value"):
                        shape.append(dim.dim_value)
                    elif dim.HasField("dim_param"):
                        shape.append(None)  # Dynamic dimension
                    else:
                        shape.append(None)
                
                shapes[name] = shape
                
                # Extract dtype
                elem_type = tensor_type.elem_type
                dtypes[name] = self._onnx_dtype_to_string(elem_type)
        
        return {
            "names": names,
            "shapes": shapes,
            "dtypes": dtypes,
        }
    
    def _calculate_resources(self, graph: onnx.GraphProto) -> ResourceEstimate:
        """Calculate resource requirements."""
        total_params = 0
        memory_bytes = 0
        
        # Count parameters from initializers (weights)
        for initializer in graph.initializer:
            tensor = numpy_helper.to_array(initializer)
            params = tensor.size
            total_params += params
            memory_bytes += tensor.nbytes
        
        return ResourceEstimate(
            parameters=total_params,
            memory_bytes=memory_bytes,
        )
    
    def _extract_metadata(self, model: onnx.ModelProto) -> Dict[str, Any]:
        """Extract metadata from model."""
        metadata = {
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "ir_version": model.ir_version,
            "opset_version": model.opset_import[0].version if model.opset_import else None,
        }
        
        # Extract custom metadata
        for prop in model.metadata_props:
            metadata[prop.key] = prop.value
        
        return metadata
    
    def _extract_attribute_value(self, attr: onnx.AttributeProto) -> Any:
        """Extract value from ONNX attribute."""
        if attr.HasField("i"):
            return attr.i
        elif attr.HasField("f"):
            return attr.f
        elif attr.HasField("s"):
            return attr.s.decode("utf-8")
        elif attr.ints:
            return list(attr.ints)
        elif attr.floats:
            return list(attr.floats)
        else:
            return None
    
    def _onnx_dtype_to_string(self, elem_type: int) -> str:
        """Convert ONNX element type to string."""
        dtype_map = {
            1: "float32",
            2: "uint8",
            3: "int8",
            4: "uint16",
            5: "int16",
            6: "int32",
            7: "int64",
            8: "string",
            9: "bool",
            10: "float16",
            11: "float64",
            12: "uint32",
            13: "uint64",
        }
        return dtype_map.get(elem_type, "unknown")
    
    def _is_standard_op(self, op_type: str) -> bool:
        """Check if operation is a standard ONNX op."""
        # List of standard ONNX operators (partial list for brevity)
        standard_ops = {
            "Conv", "Relu", "MaxPool", "Add", "Mul", "Gemm", "BatchNormalization",
            "Dropout", "Softmax", "Reshape", "Transpose", "Concat", "Split",
            "MatMul", "Sigmoid", "Tanh", "LayerNormalization", "Attention",
            "LSTM", "GRU", "Clip", "Pad", "Slice", "Gather", "Cast",
        }
        return op_type in standard_ops
    
    def _infer_model_type(self, operations: List[OpInfo]) -> ModelType:
        """Infer model type from operations."""
        op_types = [op.type for op in operations]
        op_type_set = set(op_types)
        
        # Check for vision models (convolutions)
        if "Conv" in op_type_set or "ConvTranspose" in op_type_set:
            return ModelType.VISION
        
        # Check for NLP models (attention, embeddings)
        if "Attention" in op_type_set or any("Embed" in op for op in op_type_set):
            return ModelType.NLP
        
        # Check for sequence models
        if "LSTM" in op_type_set or "GRU" in op_type_set:
            return ModelType.NLP
        
        return ModelType.UNKNOWN

