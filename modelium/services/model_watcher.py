"""
Model Watcher Service

Watches directories for new models and triggers analysis/loading.
"""

import os
import time
import threading
import logging
from pathlib import Path
from typing import List, Callable, Set

from modelium.services.model_registry import ModelRegistry, ModelStatus
from modelium.core.analyzers import FrameworkDetector, ModelAnalyzer

logger = logging.getLogger(__name__)


class ModelWatcher:
    """
    Watches directories for new model files and triggers processing.
    
    Runs in background thread and periodically scans for new models.
    """
    
    def __init__(
        self,
        watch_directories: List[str],
        scan_interval: int = 30,
        supported_formats: List[str] = None,
        on_model_discovered: Callable = None,
    ):
        """
        Initialize model watcher.
        
        Args:
            watch_directories: Directories to watch for models
            scan_interval: Seconds between scans
            supported_formats: File extensions to watch (.pt, .onnx, etc.)
            on_model_discovered: Callback when new model found
        """
        self.watch_directories = [Path(d) for d in watch_directories]
        self.scan_interval = scan_interval
        # Universal model format support
        self.supported_formats = supported_formats or [
            ".pt", ".pth", ".pth.tar",  # PyTorch
            ".onnx", ".onnx.gz",  # ONNX
            ".safetensors",  # HuggingFace SafeTensors
            ".bin",  # Generic binary
            ".ckpt", ".checkpoint",  # Checkpoints
            ".pb",  # TensorFlow
            ".tflite",  # TensorFlow Lite
            ".h5", ".hdf5",  # Keras/HDF5
            ".mlmodel",  # Core ML
            ".engine",  # TensorRT
            ".plan",  # TensorRT
        ]
        self.on_model_discovered = on_model_discovered
        
        self.registry = ModelRegistry()
        self.detector = FrameworkDetector()
        self.analyzer = ModelAnalyzer()
        
        self._seen_files: Set[str] = set()
        self._running = False
        self._thread = None
    
    def start(self):
        """Start watching directories."""
        if self._running:
            logger.warning("Model watcher already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info(f"Model watcher started, watching {len(self.watch_directories)} directories")
    
    def stop(self):
        """Stop watching directories."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Model watcher stopped")
    
    def _watch_loop(self):
        """Main watch loop (runs in background thread)."""
        while self._running:
            try:
                self._scan_directories()
            except Exception as e:
                logger.error(f"Error in watch loop: {e}")
            
            time.sleep(self.scan_interval)
    
    def _scan_directories(self):
        """Scan watch directories for new models."""
        for watch_dir in self.watch_directories:
            if not watch_dir.exists():
                logger.warning(f"Watch directory doesn't exist: {watch_dir}")
                continue
            
            try:
                # Check if watch directory itself is a model (has config.json)
                if (watch_dir / "config.json").exists():
                    watch_str = str(watch_dir)
                    if watch_str not in self._seen_files:
                        # Use directory name as model name
                        model_name = watch_dir.name if watch_dir.name else "model"
                        logger.info(f"📁 Watch directory itself is a model: {model_name} at {watch_dir}")
                        self._process_directory(watch_dir)
                
                # Scan subdirectories and files
                for item_path in watch_dir.iterdir():
                    # Check for HuggingFace model subdirectories (have config.json)
                    if item_path.is_dir() and (item_path / "config.json").exists():
                        self._process_directory(item_path)
                    # Check for individual model files (.pt, .onnx, etc.)
                    elif item_path.is_file() and self._is_model_file(item_path):
                        # Only process if watch dir itself wasn't already processed
                        if not (watch_dir / "config.json").exists():
                            self._process_file(item_path)
            except Exception as e:
                logger.error(f"Error scanning {watch_dir}: {e}")
    
    def _is_model_file(self, path: Path) -> bool:
        """Check if file is a supported model format."""
        return path.suffix.lower() in self.supported_formats
    
    def _process_directory(self, dir_path: Path):
        """Process a discovered model directory (e.g., HuggingFace model)."""
        dir_str = str(dir_path)
        
        # Skip if already seen
        if dir_str in self._seen_files:
            return
        
        self._seen_files.add(dir_str)
        
        # Use directory name, or "model" if it's the watch directory itself
        if dir_path.name:
            model_name = dir_path.name
        else:
            model_name = "model"
        
        logger.info(f"📁 Discovered model directory: {model_name} at {dir_path}")
        
        # Register in registry
        model = self.registry.register_model(model_name, dir_str)
        
        # Analyze model in background
        threading.Thread(
            target=self._analyze_model,
            args=(model_name, dir_str),
            daemon=True
        ).start()
        
        # Trigger callback (orchestrator expects directory path)
        if self.on_model_discovered:
            try:
                logger.info(f"   🔔 Calling orchestrator callback for {model_name}...")
                self.on_model_discovered(model_name, dir_str)
                logger.info(f"   ✅ Orchestrator callback completed for {model_name}")
            except Exception as e:
                logger.error(f"   ❌ Error in model discovered callback for {model_name}: {e}", exc_info=True)
    
    def _process_file(self, file_path: Path):
        """Process a discovered model file (e.g., .pt, .onnx)."""
        file_str = str(file_path)
        
        # Skip if already seen
        if file_str in self._seen_files:
            return
        
        self._seen_files.add(file_str)
        model_name = file_path.stem  # filename without extension
        
        logger.info(f"📁 Discovered model file: {model_name} at {file_path}")
        
        # Register in registry
        model = self.registry.register_model(model_name, file_str)
        
        # Analyze model in background
        threading.Thread(
            target=self._analyze_model,
            args=(model_name, file_str),
            daemon=True
        ).start()
        
        # Trigger callback
        if self.on_model_discovered:
            try:
                logger.info(f"   🔔 Calling orchestrator callback for {model_name}...")
                self.on_model_discovered(model_name, file_str)
                logger.info(f"   ✅ Orchestrator callback completed for {model_name}")
            except Exception as e:
                logger.error(f"   ❌ Error in model discovered callback for {model_name}: {e}", exc_info=True)
    
    def _analyze_model(self, model_name: str, file_path: str):
        """Analyze a model file (runs in background thread)."""
        try:
            logger.info(f"🔍 Analyzing {model_name}...")
            self.registry.update_model(model_name, status=ModelStatus.ANALYZING)
            
            # Convert string path to Path object
            path = Path(file_path)
            
            # Detect framework
            framework, _ = self.detector.detect(path)
            logger.info(f"   Framework: {framework}")
            
            # Analyze model (ModelAnalyzer.analyze expects Path, not framework)
            descriptor = self.analyzer.analyze(path, model_name=model_name)
            
            # Update registry with analysis results
            parameters = 0
            if descriptor and descriptor.resources:
                parameters = descriptor.resources.parameters
            
            self.registry.update_model(
                model_name,
                status=ModelStatus.UNLOADED,
                framework=framework.value if framework else "unknown",
                model_type=descriptor.model_type.value if descriptor and descriptor.model_type else "unknown",
                size_bytes=os.path.getsize(path),
                parameters=parameters,
            )
            
            logger.info(f"   ✅ Analysis complete: {model_name}")
            
        except Exception as e:
            logger.error(f"   ❌ Analysis failed for {model_name}: {e}")
            self.registry.update_model(
                model_name,
                status=ModelStatus.ERROR,
                error=str(e)
            )

