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
        self.supported_formats = supported_formats or [
            ".pt", ".pth", ".onnx", ".safetensors", ".bin"
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
                for file_path in watch_dir.iterdir():
                    if file_path.is_file() and self._is_model_file(file_path):
                        self._process_file(file_path)
            except Exception as e:
                logger.error(f"Error scanning {watch_dir}: {e}")
    
    def _is_model_file(self, path: Path) -> bool:
        """Check if file is a supported model format."""
        return path.suffix.lower() in self.supported_formats
    
    def _process_file(self, file_path: Path):
        """Process a discovered model file."""
        file_str = str(file_path)
        
        # Skip if already seen
        if file_str in self._seen_files:
            return
        
        self._seen_files.add(file_str)
        model_name = file_path.stem  # filename without extension
        
        logger.info(f"📁 Discovered model: {model_name} at {file_path}")
        
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
                self.on_model_discovered(model_name, file_str)
            except Exception as e:
                logger.error(f"Error in model discovered callback: {e}")
    
    def _analyze_model(self, model_name: str, file_path: str):
        """Analyze a model file (runs in background thread)."""
        try:
            logger.info(f"🔍 Analyzing {model_name}...")
            self.registry.update_model(model_name, status=ModelStatus.ANALYZING)
            
            # Detect framework
            framework = self.detector.detect(file_path)
            logger.info(f"   Framework: {framework}")
            
            # Analyze model
            descriptor = self.analyzer.analyze(file_path, framework)
            
            # Update registry with analysis results
            self.registry.update_model(
                model_name,
                status=ModelStatus.UNLOADED,
                framework=framework,
                model_type=descriptor.model_type if descriptor else "unknown",
                size_bytes=os.path.getsize(file_path),
                parameters=descriptor.total_parameters if descriptor else 0,
            )
            
            logger.info(f"   ✅ Analysis complete: {model_name}")
            
        except Exception as e:
            logger.error(f"   ❌ Analysis failed for {model_name}: {e}")
            self.registry.update_model(
                model_name,
                status=ModelStatus.ERROR,
                error=str(e)
            )

