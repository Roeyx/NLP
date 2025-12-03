"""
Emotion Detection Project - Source Package

A modular package for emotion classification using Bidirectional GRU with GloVe embeddings.

Components:
    - config: Configuration constants and hyperparameters
    - utils: Utility functions (timing, GPU setup, directory management)
    - preprocessing: Text preprocessing and tokenization
    - embeddings: GloVe embedding loader with caching
    - model: EmotionGRU model (tf.keras.Model subclass)
    - trainer: Training pipeline with grid search

Example Usage:
    from src import GRUTrainer
    
    trainer = GRUTrainer()
    model, params = trainer.run_full_pipeline()
"""

from .config import (
    MAX_LEN,
    EMBED_DIM,
    LABEL_MAP,
    NUM_CLASSES,
    PARAM_GRID,
    DATA_DIR,
    OUTPUT_DIR,
)
from .utils import (
    timing_decorator,
    setup_gpu,
    ensure_dirs,
    setup_environment,
)
from .preprocessing import TextPreprocessor
from .embeddings import EmbeddingLoader
from .model import EmotionGRU, load_model
from .trainer import GRUTrainer

__all__ = [
    # Config
    'MAX_LEN',
    'EMBED_DIM',
    'LABEL_MAP',
    'NUM_CLASSES',
    'PARAM_GRID',
    'DATA_DIR',
    'OUTPUT_DIR',
    # Utils
    'timing_decorator',
    'setup_gpu',
    'ensure_dirs',
    'setup_environment',
    # Classes
    'TextPreprocessor',
    'EmbeddingLoader',
    'EmotionGRU',
    'GRUTrainer',
    # Functions
    'load_model',
]

__version__ = '1.0.0'
