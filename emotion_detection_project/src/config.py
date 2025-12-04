"""
Configuration constants and hyperparameters for the Emotion Detection project.
"""

import os

# =============================================================================
# Directory Configuration
# =============================================================================

# Base directories (relative to project root)
DATA_DIR = "data"
OUTPUT_DIR = "outputs"

# Output subdirectories
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# =============================================================================
# Data Paths
# =============================================================================

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "validation.csv")

# GloVe embeddings (relative to project root - large external file)
GLOVE_FILE = "glove.twitter.27B.200d.txt"

# =============================================================================
# Model Hyperparameters
# =============================================================================

MAX_LEN = 50
EMBED_DIM = 200

# =============================================================================
# Training Configuration
# =============================================================================

# Fixed batch size
BATCH_SIZE = 1024

# Grid search epochs (fast evaluation)
GRID_WARMUP_EPOCHS = 3
GRID_FINETUNE_EPOCHS = 6

# Final training epochs
FINAL_WARMUP_EPOCHS = 5
FINAL_FINETUNE_EPOCHS = 15

# =============================================================================
# Label Mapping
# =============================================================================

LABEL_MAP = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

NUM_CLASSES = len(LABEL_MAP)

# =============================================================================
# Hyperparameter Grid for Grid Search (reduced & LR-focused)
# =============================================================================

PARAM_GRID = {
    # Architecture (fixed to reduce search space)
    'gru_units': [256, 512],
    'dense_units': [128, 256],
    
    # Regularization (fixed / minimal)
    'dropout': [0.3],
    'spatial_dropout': [0.2],
    'recurrent_dropout': [0.0],
    
    # Optimization (focus of the search)
    # warmup_lr = LR used while embeddings are frozen
    # fine_tune_lr = LR used after unfreezing embeddings
    'warmup_lr': [1e-3, 5e-3, 1e-2],
    'fine_tune_lr': [5e-4, 1e-3],
}

# Total combinations: 2 * 2 * 1 * 1 * 1 * 3 * 2 = 24

# =============================================================================
# Output File Paths
# =============================================================================

# Cache files
EMBEDDING_CACHE_PATH = os.path.join(CACHE_DIR, "embedding_matrix.npz")

# Model files
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "final_model.keras")

# Result files
GRID_RESULTS_PATH = os.path.join(RESULTS_DIR, "grid_results.json")
BEST_PARAMS_PATH = os.path.join(RESULTS_DIR, "best_params.json")
WORD_INDEX_PATH = os.path.join(RESULTS_DIR, "word_index.pkl")
CLASSIFICATION_REPORT_PATH = os.path.join(RESULTS_DIR, "classification_report.txt")
