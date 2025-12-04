# Emotion Detection Project

A modular deep learning system for multi-class emotion classification using Bidirectional GRU with pre-trained GloVe embeddings.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Module Documentation](#module-documentation)
- [Data Flow](#data-flow)
- [Usage](#usage)
- [Output Files](#output-files)
- [Configuration](#configuration)

---

## Overview

This project implements a text-based emotion classifier that categorizes input text into 6 emotion categories:
- **Sadness** (0)
- **Joy** (1)
- **Love** (2)
- **Anger** (3)
- **Fear** (4)
- **Surprise** (5)

The system uses a **freeze-thaw training strategy** where embeddings are initially frozen during warmup, then unfrozen for fine-tuning, optimizing both training speed and model performance.

---

## Project Structure

```
emotion_detection_project/
├── main.py                      # Entry point - orchestrates the pipeline
├── glove.twitter.27B.200d.txt   # Pre-trained GloVe embeddings (200d)
├── data/
│   ├── train.csv                # Training data (text, label)
│   └── validation.csv           # Validation data (text, label)
├── src/
│   ├── __init__.py              # Package initialization & exports
│   ├── config.py                # Centralized configuration & hyperparameters
│   ├── utils.py                 # Utility functions (GPU setup, timing, etc.)
│   ├── preprocessing.py         # Text cleaning & tokenization
│   ├── embeddings.py            # GloVe embedding loader with caching
│   ├── model.py                 # EmotionGRU model architecture
│   └── trainer.py               # Training pipeline & grid search
└── outputs/
    ├── cache/
    │   └── embedding_matrix.npz # Cached embedding matrix
    ├── models/
    │   └── final_model.keras    # Trained model
    └── results/
        ├── grid_results.json    # All grid search results
        ├── best_params.json     # Best hyperparameters found
        ├── word_index.pkl       # Vocabulary mapping
        └── classification_report.txt  # Final evaluation metrics
```

---

## Module Documentation

### 1. `main.py` - Entry Point

**Purpose:** Orchestrates the complete training pipeline from start to finish.

**Key Functions:**
- `main()`: Sets up environment, creates directories, initializes trainer, runs pipeline

**Workflow:**
```python
1. Set up environment variables (LD_LIBRARY_PATH)
2. Import TensorFlow and project modules
3. Create output directories
4. Initialize GRUTrainer
5. Run full training pipeline
6. Return trained model and best parameters
```

---

### 2. `src/config.py` - Configuration Hub

**Purpose:** Centralized configuration for all hyperparameters, paths, and settings.

**Key Constants:**

**Directory Paths:**
- `DATA_DIR`, `OUTPUT_DIR`, `CACHE_DIR`, `MODELS_DIR`, `RESULTS_DIR`

**Data Paths:**
- `TRAIN_PATH`, `VAL_PATH`, `GLOVE_FILE`

**Model Hyperparameters:**
- `MAX_LEN = 50` - Maximum sequence length
- `EMBED_DIM = 200` - Embedding dimension
- `BATCH_SIZE = 1024` - Training batch size

**Training Configuration:**
- `GRID_WARMUP_EPOCHS = 3` - Warmup epochs for grid search
- `GRID_FINETUNE_EPOCHS = 6` - Finetune epochs for grid search
- `FINAL_WARMUP_EPOCHS = 5` - Warmup epochs for final training
- `FINAL_FINETUNE_EPOCHS = 15` - Finetune epochs for final training

**Label Mapping:**
- `LABEL_MAP = {0: 'sadness', 1: 'joy', ...}` - Maps indices to emotion names
- `NUM_CLASSES = 6` - Number of emotion categories

**Hyperparameter Grid:**
```python
PARAM_GRID = {
    'gru_units': [256, 512],           # GRU hidden units
    'dense_units': [128, 256],         # Dense layer units
    'dropout': [0.3],                  # Dropout rate
    'spatial_dropout': [0.2],          # Spatial dropout rate
    'recurrent_dropout': [0.0],        # Recurrent dropout rate
    'warmup_lr': [1e-3, 5e-3, 1e-2],  # Warmup learning rates
    'fine_tune_lr': [5e-4, 1e-3],     # Fine-tune learning rates
}
# Total: 24 combinations
```

---

### 3. `src/utils.py` - Utility Functions

**Purpose:** Provides helper functions for timing, GPU setup, and directory management.

**Key Functions:**

#### `timing_decorator(func)`
- **Type:** Decorator
- **Purpose:** Measures and prints execution time of functions
- **Usage:** `@timing_decorator` above function definitions

#### `setup_gpu()`
- **Returns:** `(has_gpu: bool, batch_size: int)`
- **Purpose:** Detects GPU, configures memory growth, returns optimized batch size
- **Behavior:**
  - GPU detected → batch_size = 1024
  - No GPU → batch_size = 32

#### `ensure_dirs()`
- **Purpose:** Creates all output directories if they don't exist
- **Creates:** `outputs/`, `outputs/cache/`, `outputs/models/`, `outputs/results/`

#### `format_time(seconds)`
- **Returns:** Formatted time string (e.g., "2m 34.56s")
- **Purpose:** Human-readable time formatting

---

### 4. `src/preprocessing.py` - Text Preprocessing

**Purpose:** Handles all text cleaning, tokenization, and sequence preparation.

**Class: `TextPreprocessor`**

**Attributes:**
- `tokenizer`: NLTK TweetTokenizer for social media text
- `word_index`: Vocabulary mapping (word → integer ID)

**Key Methods:**

#### `preprocess_text(text)` [static]
- **Input:** Raw text string
- **Output:** Cleaned text
- **Steps:**
  1. Convert to lowercase
  2. Remove non-alphabetic characters
  3. Normalize whitespace
- **Example:** `"Hello!!! World 123" → "hello world"`

#### `tokenize_text(text_list)`
- **Input:** List of text strings
- **Output:** List of token lists
- **Example:** `["hello world"] → [["hello", "world"]]`

#### `build_vocab(tokenized_texts)`
- **Input:** List of token lists
- **Output:** Word index dictionary
- **Special Tokens:**
  - `<PAD>` (0) - Padding token
  - `<UNK>` (1) - Unknown token
- **Behavior:** Assigns unique integer ID to each unique word

#### `text_to_sequences(tokenized_texts, word_index)`
- **Input:** Token lists, vocabulary
- **Output:** Integer sequences
- **Example:** `[["hello", "world"]] → [[45, 67]]`

#### `load_and_prep_data(filepath, word_index=None, is_train=True)`
- **Input:** CSV path, optional vocabulary, train flag
- **Output:** `(X, labels, word_index)` where X is padded sequences
- **Pipeline:**
  1. Load CSV (expects 'text' and 'label' columns)
  2. Preprocess text
  3. Tokenize
  4. Build/use vocabulary
  5. Convert to sequences
  6. Pad sequences to MAX_LEN

---

### 5. `src/embeddings.py` - Embedding Management

**Purpose:** Loads pre-trained GloVe embeddings with intelligent caching.

**Class: `EmbeddingLoader`**

**Key Methods:**

#### `load_glove_matrix(word_index, embed_dim=200, glove_path, cache_path)` [static]
- **Input:** Vocabulary, embedding dimension, file paths
- **Output:** Embedding matrix (numpy array)
- **Shape:** `(vocab_size, embed_dim)`
- **Workflow:**
  1. Check if cache exists and vocabulary matches
  2. If cache hit → load from cache (fast)
  3. If cache miss → build from GloVe file
  4. Save to cache for future runs
- **Cache Benefits:** 
  - First run: ~30-60 seconds
  - Subsequent runs: ~1-2 seconds

#### `_build_embedding_matrix(word_index, embed_dim, glove_path)` [static, internal]
- **Purpose:** Constructs embedding matrix from scratch
- **Steps:**
  1. Parse GloVe text file (1.2M+ words)
  2. Create embeddings_index dictionary
  3. Initialize embedding_matrix with zeros
  4. For each word in vocabulary:
     - If in GloVe → use pre-trained vector
     - If OOV → random initialization (normal distribution)
  5. Report coverage statistics

#### `_save_cache(embedding_matrix, word_index, cache_path)` [static, internal]
- **Purpose:** Saves embedding matrix and vocabulary to compressed file
- **Format:** `.npz` (NumPy compressed format)
- **Contents:** embedding_matrix, word_index

---

### 6. `src/model.py` - Neural Network Architecture

**Purpose:** Defines the EmotionGRU model using TensorFlow/Keras.

**Class: `EmotionGRU(tf.keras.Model)`**

**Architecture:**
```
Input (batch_size, MAX_LEN)
    ↓
Embedding Layer (vocab_size, 200) [pre-trained GloVe, initially frozen]
    ↓
SpatialDropout1D (0.2)
    ↓
Bidirectional GRU (units × 2) [forward + backward]
    ↓
Dense Layer (dense_units, ReLU activation)
    ↓
Dropout (dropout_rate)
    ↓
Output Dense (6 classes, Softmax)
```

**Constructor Parameters:**
- `vocab_size`: Size of vocabulary
- `embedding_matrix`: Pre-trained GloVe weights
- `gru_units`: Number of GRU units (256 or 512)
- `dense_units`: Dense layer size (128 or 256)
- `dropout`: Dropout rate (0.3)
- `spatial_dropout`: Spatial dropout rate (0.2)
- `recurrent_dropout`: Recurrent dropout rate (0.0)

**Key Methods:**

#### `call(inputs, training=None)`
- **Purpose:** Forward pass through the network
- **Input:** Tokenized sequences
- **Output:** Class probabilities (6 classes)

#### `freeze_embeddings()`
- **Purpose:** Makes embedding layer non-trainable
- **Use Case:** Warmup phase - train other layers first

#### `unfreeze_embeddings()`
- **Purpose:** Makes embedding layer trainable
- **Use Case:** Fine-tuning phase - adapt embeddings to task

#### `train_freeze_thaw(X_train, y_train, X_val, y_val, warmup_epochs, finetune_epochs, warmup_lr, fine_tune_lr, batch_size)`
- **Purpose:** Two-phase training strategy
- **Returns:** `(history1, history2)` - Training histories for both phases

**Phase 1 - Warmup (Frozen Embeddings):**
1. Freeze embedding layer
2. Compile with `warmup_lr`
3. Train for `warmup_epochs`
4. → Quick convergence on task-specific patterns

**Phase 2 - Fine-tuning (Unfrozen Embeddings):**
1. Unfreeze embedding layer
2. Compile with `fine_tune_lr` (typically lower)
3. Train for `finetune_epochs` with callbacks:
   - `EarlyStopping`: Stops if val_accuracy plateaus (patience=3)
   - `ReduceLROnPlateau`: Reduces LR if val_loss plateaus (patience=2)
4. → Adapts embeddings to emotion classification

**Additional Methods:**
- `get_config()`: Serialization support
- `from_config(config)`: Deserialization support
- `summary_str()`: Human-readable architecture summary

**Module Function:**
#### `load_model(filepath)`
- **Purpose:** Loads saved EmotionGRU model from disk
- **Returns:** Loaded Keras model
- **Handles:** Custom object registration

---

### 7. `src/trainer.py` - Training Pipeline & Grid Search

**Purpose:** Orchestrates the complete training workflow including hyperparameter search.

**Class: `GRUTrainer`**

**Attributes:**
- `param_grid`: Hyperparameter search space
- `preprocessor`: TextPreprocessor instance
- `best_params`: Best hyperparameters from grid search
- `best_model`: Best trained model
- `batch_size`: Training batch size
- `has_gpu`: GPU availability flag

**Complete Training Pipeline:**

```
1. load_data() → Preprocess train & validation data
           ↓
2. load_glove_matrix() → Build embedding matrix
           ↓
3. run_grid_search() → Test all hyperparameter combinations
           ↓
4. train_final_model() → Train with best params & more epochs
           ↓
5. evaluate_and_report() → Generate classification report
           ↓
6. Save artifacts → Model, params, vocabulary, results
```

**Key Methods:**

#### `load_data(train_path, val_path)`
- **Returns:** `(X_train, y_train, X_val, y_val, word_index)`
- **Process:**
  1. Load training data (builds vocabulary)
  2. Load validation data (uses training vocabulary)
  3. Print dataset statistics

#### `run_grid_search(X_train, y_train, X_val, y_val, embedding_matrix, vocab_size)`
- **Purpose:** Exhaustive hyperparameter search
- **Strategy:** Tests all combinations in PARAM_GRID (24 total)
- **For Each Configuration:**
  1. Create model with params
  2. Train with freeze-thaw (3+6 epochs)
  3. Record validation accuracy & time
  4. Append result to `grid_results.json` (incremental save)
  5. Update best_params if better accuracy
  6. Save best_params to `best_params.json`
- **Returns:** Best hyperparameters dictionary
- **Error Handling:** Continues on failure, raises error if all fail

#### `train_final_model(X_train, y_train, X_val, y_val, embedding_matrix, vocab_size, params=None)`
- **Purpose:** Train final model with best hyperparameters
- **Uses:** More epochs than grid search (5+15 vs 3+6)
- **Default Behavior:** Uses `self.best_params` if params not provided
- **Saves:** Trained model to `outputs/models/final_model.keras`
- **Returns:** `(model, history1, history2)`

#### `evaluate_and_report(model, X_val, y_val)`
- **Purpose:** Generate detailed evaluation metrics
- **Output:**
  - Prints classification report to console
  - Saves report to `classification_report.txt`
- **Metrics:** Precision, recall, F1-score per class
- **Returns:** Predicted labels array

#### `run_full_pipeline(train_path, val_path)`
- **Purpose:** Executes complete training workflow
- **Steps:**
  1. Load and preprocess data
  2. Build embedding matrix
  3. Run grid search
  4. Train final model
  5. Evaluate and report
  6. Save vocabulary
  7. Print summary
- **Returns:** `(final_model, best_params)`

**Helper Methods:**

#### `_create_model(vocab_size, embedding_matrix, params)`
- **Purpose:** Factory method for model instantiation
- **Returns:** Configured EmotionGRU instance

#### `_append_result_to_json(params, val_accuracy, time)`
- **Purpose:** Incrementally saves grid search results
- **Behavior:**
  1. Load existing `grid_results.json` (or create empty list)
  2. Append new result
  3. Save back to file
- **Format:** List of dicts with keys: `params`, `val_accuracy`, `time`

#### `_save_best_params(params)`
- **Purpose:** Saves current best hyperparameters
- **File:** `best_params.json`
- **Updated:** Each time a better configuration is found

#### `_save_word_index(word_index)`
- **Purpose:** Saves vocabulary for inference
- **File:** `word_index.pkl` (pickle format)
- **Usage:** Required for preprocessing new text at inference time

#### `_save_classification_report(report)`
- **Purpose:** Saves evaluation metrics as text
- **File:** `classification_report.txt`

#### `_print_summary()`
- **Purpose:** Prints list of all saved files
- **Output:** Formatted summary of training artifacts

---

## Data Flow

### Training Flow Diagram

```
main.py
  ↓
GRUTrainer.__init__()
  ↓
run_full_pipeline()
  │
  ├─→ load_data()
  │     ├─→ TextPreprocessor.load_and_prep_data(train.csv)
  │     │     ├─→ preprocess_text() → Clean text
  │     │     ├─→ tokenize_text() → Split into tokens
  │     │     ├─→ build_vocab() → Create word_index
  │     │     └─→ text_to_sequences() → Convert to integers
  │     │
  │     └─→ TextPreprocessor.load_and_prep_data(validation.csv)
  │           └─→ Use existing word_index
  │
  ├─→ EmbeddingLoader.load_glove_matrix()
  │     ├─→ Check cache
  │     ├─→ Load GloVe file if needed
  │     └─→ Build embedding_matrix (vocab_size × 200)
  │
  ├─→ run_grid_search()
  │     └─→ For each hyperparameter combination:
  │           ├─→ _create_model() → EmotionGRU instance
  │           ├─→ model.train_freeze_thaw()
  │           │     ├─→ Phase 1: Frozen embeddings (3 epochs)
  │           │     └─→ Phase 2: Unfrozen embeddings (6 epochs)
  │           ├─→ Evaluate validation accuracy
  │           ├─→ _append_result_to_json() → Save result
  │           └─→ _save_best_params() → Update if best
  │
  ├─→ train_final_model()
  │     ├─→ _create_model() with best_params
  │     ├─→ model.train_freeze_thaw()
  │     │     ├─→ Phase 1: Frozen embeddings (5 epochs)
  │     │     └─→ Phase 2: Unfrozen embeddings (15 epochs)
  │     └─→ model.save() → final_model.keras
  │
  ├─→ evaluate_and_report()
  │     ├─→ model.predict() on validation data
  │     └─→ _save_classification_report()
  │
  ├─→ _save_word_index() → word_index.pkl
  │
  └─→ _print_summary() → Display results
```

---

## Usage

### Basic Usage

```bash
# From project root
cd emotion_detection_project
python main.py
```

### Expected Output

```
============================================================
Bidirectional GRU Emotion Classification
============================================================
TensorFlow version: 2.x.x
Python: 3.x.x
✓ Output directories ready
✅ GPU Detected: 1 device(s)
✓ Set BATCH_SIZE = 1024 (GPU-optimized)

=== Loading Data ===
Loading data/train.csv...
Loading data/validation.csv...
Training samples: 16000
Validation samples: 2000
Vocabulary size: 12543

=== Building Embedding Matrix ===
Loading cached embedding matrix from outputs/cache/embedding_matrix.npz...
✓ Loaded cached embedding matrix: shape (12543, 200)

=== Grid Search: 24 combinations ===
Batch size: 1024
⏳ Note: First iteration will be slow due to GPU warmup/compilation...

[1/24] Testing: {'gru_units': 256, 'dense_units': 128, ...}
Phase 1: Warmup training (lr=0.001)...
Phase 2: Fine-tuning (lr=0.0005)...
  ✓ Val Acc: 0.8450 (took 45.3s)
  → Appended result to outputs/results/grid_results.json
  *** New best! ***
  ✓ Best params saved to outputs/results/best_params.json

[2/24] Testing: ...
...

==================================================
BEST RESULT: 0.8725
BEST PARAMS: {'gru_units': 512, 'dense_units': 256, ...}
==================================================
⏱️  run_grid_search completed in 18m 23.45s

=== Training Final Model ===
Parameters: {'gru_units': 512, 'dense_units': 256, ...}
...
✓ Final model saved to outputs/models/final_model.keras

=== Final Evaluation ===
Classification Report:
              precision    recall  f1-score   support
     sadness       0.87      0.89      0.88       350
         joy       0.92      0.91      0.91       400
...

=== DONE ===
Saved files:
  - outputs/models/final_model.keras (trained model)
  - outputs/results/best_params.json (best hyperparameters)
  - outputs/results/grid_results.json (all grid search results)
  - outputs/results/word_index.pkl (vocabulary)
  - outputs/results/classification_report.txt (evaluation report)
```

---

## Output Files

### 1. `outputs/models/final_model.keras`
- **Format:** Keras SavedModel
- **Contents:** Complete trained model (architecture + weights)
- **Usage:** Load with `tf.keras.models.load_model()` or `load_model()` from model.py
- **Size:** ~50-100 MB

### 2. `outputs/results/best_params.json`
- **Format:** JSON
- **Contents:** Best hyperparameters from grid search
- **Example:**
```json
{
  "gru_units": 512,
  "dense_units": 256,
  "dropout": 0.3,
  "spatial_dropout": 0.2,
  "recurrent_dropout": 0.0,
  "warmup_lr": 0.005,
  "fine_tune_lr": 0.0005
}
```

### 3. `outputs/results/grid_results.json`
- **Format:** JSON (list of dicts)
- **Contents:** All grid search experiments
- **Updates:** Appended incrementally after each iteration
- **Example:**
```json
[
  {
    "params": {"gru_units": 256, "dense_units": 128, ...},
    "val_accuracy": 0.8450,
    "time": 45.3
  },
  {
    "params": {"gru_units": 512, "dense_units": 128, ...},
    "val_accuracy": 0.8725,
    "time": 52.1
  }
]
```

### 4. `outputs/results/word_index.pkl`
- **Format:** Python pickle
- **Contents:** Vocabulary dictionary (word → integer ID)
- **Usage:** Required for preprocessing new text during inference
- **Example:** `{"<PAD>": 0, "<UNK>": 1, "hello": 2, "world": 3, ...}`

### 5. `outputs/results/classification_report.txt`
- **Format:** Plain text
- **Contents:** Precision, recall, F1-score per class
- **Generated by:** sklearn.metrics.classification_report

### 6. `outputs/cache/embedding_matrix.npz`
- **Format:** NumPy compressed archive
- **Contents:** Embedding matrix + word_index
- **Purpose:** Cache to avoid reloading GloVe file
- **Invalidation:** Automatically rebuilt if vocabulary changes

---

## Configuration

### Modifying Hyperparameters

Edit `src/config.py` to adjust:

**Model Architecture:**
```python
PARAM_GRID = {
    'gru_units': [128, 256, 512],      # Add/remove values
    'dense_units': [64, 128, 256],
    'dropout': [0.2, 0.3, 0.5],
}
```

**Training Epochs:**
```python
GRID_WARMUP_EPOCHS = 3      # Fast evaluation
GRID_FINETUNE_EPOCHS = 6

FINAL_WARMUP_EPOCHS = 5     # More thorough training
FINAL_FINETUNE_EPOCHS = 15
```

**Sequence Length:**
```python
MAX_LEN = 50  # Increase for longer texts
```

**Batch Size:**
```python
BATCH_SIZE = 1024  # Reduce if GPU memory limited
```

### Using Different Embeddings

1. Download different GloVe dimension (50d, 100d, 300d)
2. Update `config.py`:
```python
GLOVE_FILE = "glove.twitter.27B.100d.txt"
EMBED_DIM = 100
```
3. Delete cache: `rm outputs/cache/embedding_matrix.npz`
4. Re-run training

---

## Advanced Usage

### Loading and Using Trained Model

```python
from src import load_model, TextPreprocessor
import pickle
import numpy as np

# Load model
model = load_model('outputs/models/final_model.keras')

# Load vocabulary
with open('outputs/results/word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)

# Preprocess new text
preprocessor = TextPreprocessor()
text = "I am so happy today!"
cleaned = preprocessor.preprocess_text(text)
tokens = preprocessor.tokenize_text([cleaned])
sequences = preprocessor.text_to_sequences(tokens, word_index)

from tensorflow.keras.preprocessing.sequence import pad_sequences
X = pad_sequences(sequences, maxlen=50, padding='post')

# Predict
predictions = model.predict(X)
emotion_idx = np.argmax(predictions, axis=1)[0]

# Map to emotion name
LABEL_MAP = {0: 'sadness', 1: 'joy', 2: 'love', 
             3: 'anger', 4: 'fear', 5: 'surprise'}
emotion = LABEL_MAP[emotion_idx]
print(f"Predicted emotion: {emotion}")
```

### Custom Training

```python
from src import GRUTrainer, PARAM_GRID

# Custom hyperparameter grid
custom_grid = {
    'gru_units': [128],
    'dense_units': [64],
    'dropout': [0.3],
    'spatial_dropout': [0.2],
    'recurrent_dropout': [0.0],
    'warmup_lr': [1e-3],
    'fine_tune_lr': [5e-4],
}

# Initialize trainer with custom grid
trainer = GRUTrainer(param_grid=custom_grid)

# Run pipeline
model, params = trainer.run_full_pipeline()
```

---

## Module Relationships

```
main.py
   │
   └─→ imports & calls: GRUTrainer, ensure_dirs
        │
        └─→ GRUTrainer (trainer.py)
             │
             ├─→ uses: TextPreprocessor (preprocessing.py)
             │         └─→ uses: config.MAX_LEN
             │
             ├─→ uses: EmbeddingLoader (embeddings.py)
             │         └─→ uses: config.EMBED_DIM, GLOVE_FILE
             │
             ├─→ uses: EmotionGRU (model.py)
             │         └─→ uses: config.NUM_CLASSES, EMBED_DIM
             │
             ├─→ uses: setup_gpu, timing_decorator (utils.py)
             │
             └─→ uses: config.*
                       - PARAM_GRID
                       - BATCH_SIZE
                       - LABEL_MAP
                       - All paths
```

**Dependency Flow:**
1. **config.py** → Imported by all modules (no dependencies)
2. **utils.py** → Imports config.py only
3. **preprocessing.py** → Imports config.py only
4. **embeddings.py** → Imports config.py, utils.py
5. **model.py** → Imports config.py
6. **trainer.py** → Imports all above modules
7. **main.py** → Imports trainer.py (and transitively, everything)

---

## Summary

This project demonstrates a **production-ready, modular machine learning pipeline** with:

✅ **Clean separation of concerns** - Each module has a single responsibility
✅ **Comprehensive documentation** - Every class and method documented
✅ **Efficient caching** - Embedding matrix cached to disk
✅ **Flexible configuration** - Easy hyperparameter tuning via config.py
✅ **Robust error handling** - Graceful failure handling in grid search
✅ **Reproducible results** - All hyperparameters and results saved
✅ **Incremental saves** - Results saved after each iteration (safe for long runs)
✅ **Two-phase training** - Freeze-thaw strategy for optimal performance

The architecture is designed to be easily extensible for different NLP classification tasks by modifying the configuration and swapping the model architecture while keeping the pipeline structure intact.
