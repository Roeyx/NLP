"""
Utility functions for the Emotion Detection project.
Includes timing decorator, GPU setup, and directory management.
"""

import os
import time
from functools import wraps

import tensorflow as tf

from .config import (
    OUTPUT_DIR,
    CACHE_DIR,
    MODELS_DIR,
    RESULTS_DIR,
)


def timing_decorator(func):
    """Decorator to measure execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        if elapsed_time < 60:
            print(f"⏱️  {func.__name__} completed in {elapsed_time:.2f} seconds")
        else:
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            print(f"⏱️  {func.__name__} completed in {minutes}m {seconds:.2f}s")
        
        return result
    return wrapper


def setup_environment():
    """Set up environment variables before TensorFlow import.
    
    Note: This should be called at the very start of the program,
    ideally in main.py before any other imports.
    """
    os.environ['LD_LIBRARY_PATH'] = (
        f"{os.environ.get('CONDA_PREFIX', '')}/lib:"
        f"{os.environ.get('LD_LIBRARY_PATH', '')}"
    )


def setup_gpu():
    """Configure GPU settings for TensorFlow.
    
    Returns:
        tuple: (has_gpu: bool, batch_size: int)
    """
    gpus = tf.config.list_physical_devices('GPU')
    has_gpu = len(gpus) > 0
    batch_size = 1024 if has_gpu else 32
    
    if has_gpu:
        print(f"✅ GPU Detected: {len(gpus)} device(s)")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"⚠️  GPU memory growth setting failed: {e}")
        print(f"✓ Set BATCH_SIZE = {batch_size} (GPU-optimized)")
    else:
        print(f"⚠️ No GPU detected")
        print(f"✓ Set BATCH_SIZE = {batch_size} (CPU-optimized)")
    
    return has_gpu, batch_size


def ensure_dirs():
    """Create all output directories if they don't exist."""
    directories = [
        OUTPUT_DIR,
        CACHE_DIR,
        MODELS_DIR,
        RESULTS_DIR,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Output directories ready")


def format_time(seconds):
    """Format seconds into a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"
