"""
Modelium Brain - Unified LLM for conversion planning and orchestration.

This module provides the "brain" of Modelium - a single fine-tuned LLM that:
1. Analyzes models and generates conversion/deployment plans
2. Makes intelligent orchestration decisions (load/unload, GPU packing)
"""

from modelium.brain.unified_brain import ModeliumBrain
from modelium.brain.prompts import CONVERSION_SYSTEM_PROMPT, ORCHESTRATION_SYSTEM_PROMPT

__all__ = ["ModeliumBrain", "CONVERSION_SYSTEM_PROMPT", "ORCHESTRATION_SYSTEM_PROMPT"]

