#!/usr/bin/env python3
"""
Emotion Detection Project - Main Entry Point

Bidirectional GRU Emotion Classification using GloVe embeddings.
This script orchestrates the complete training pipeline.

Usage:
    python main.py

Output:
    - outputs/models/final_model.keras              (trained model)
    - outputs/results/best_params.json              (best hyperparameters)
    - outputs/results/grid_results.json             (all grid search results)
    - outputs/results/word_index.pkl                (vocabulary)
    - outputs/results/classification_report.txt     (evaluation)
"""

import os
import sys

# CRITICAL: Set LD_LIBRARY_PATH before importing TensorFlow
os.environ['LD_LIBRARY_PATH'] = (
    f"{os.environ.get('CONDA_PREFIX', '')}/lib:"
    f"{os.environ.get('LD_LIBRARY_PATH', '')}"
)

import tensorflow as tf

from src import GRUTrainer, ensure_dirs


def main():
    """Main entry point for the emotion detection training pipeline."""
    print("=" * 60)
    print("Bidirectional GRU Emotion Classification")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python: {sys.version.split()[0]}")
    
    # Ensure all output directories exist
    ensure_dirs()
    
    # Run the training pipeline
    trainer = GRUTrainer()
    model, params = trainer.run_full_pipeline()
    
    return model, params


if __name__ == "__main__":
    model, params = main()
