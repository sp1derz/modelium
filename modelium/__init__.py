"""
Modelium - Automated ML Model Ingestion & Deployment Pipeline

A comprehensive system for analyzing, converting, optimizing, and deploying
arbitrary ML models to production inference infrastructure.

Supports: vLLM, Ray Serve, TensorRT, Triton
"""

__version__ = "0.1.0"
__author__ = "Modelium Team"
__license__ = "Apache-2.0"

from modelium.core.descriptor import ModelDescriptor
from modelium.core.analyzers import FrameworkDetector
from modelium.config import get_config, load_config, ModeliumConfig
from modelium.brain import ModeliumBrain

__all__ = [
    "ModelDescriptor",
    "FrameworkDetector",
    "get_config",
    "load_config",
    "ModeliumConfig",
    "ModeliumBrain",
]

