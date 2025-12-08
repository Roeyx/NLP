# Emotion Detection Project (LSTM & GRU)

This repository implements an emotion classification pipeline using pretrained GloVe embeddings and recurrent neural networks (LSTM and GRU). The project compares two approaches:

- **LSTM.py**: Baseline single-phase training with comprehensive grid search
- **GRU.py**: Advanced two-phase training pipeline with freeze-thaw optimization

## Main Features

- Tokenization using Keras `Tokenizer` with OOV handling
- GloVe Twitter embeddings (100d) for semantic understanding
- Automatic GloVe download via `kagglehub`
- Comprehensive hyperparameter grid search
- Two-phase training strategy (GRU): baseline comparison + freeze-thaw optimization
- Spatial dropout for improved regularization in NLP tasks
- EarlyStopping and ReduceLROnPlateau callbacks for robust training

---

## Table of Contents
- [Project Structure](#project-structure)
- [Requirements & Setup](#requirements--setup)
- [Data & GloVe Embeddings](#data--glove-embeddings)
- [LSTM Workflow](#lstm-workflow)
- [GRU Workflow (Two-Phase Pipeline)](#gru-workflow-two-phase-pipeline)
- [Hyperparameter Grids](#hyperparameter-grids)
- [Examples & Usage](#examples--usage)
- [Tips & Troubleshooting](#tips--troubleshooting)
- [Developer Notes](#developer-notes)

---

## Project Structure

```
NLP/
├── GRU.py                    # Two-phase GRU pipeline (Phase 1: baseline, Phase 2: optimization)
├── LSTM.py                   # Baseline LSTM with comprehensive grid search
├── train.csv                 # Training data (text, label)
├── validation.csv            # Validation data (text, label)
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── glove.twitter.27B.*.txt   # GloVe embeddings (auto-downloaded)
```

### Key Files

- **`GRU.py`**: Advanced pipeline with `GRUModel` and `GRUTrainer` classes
  - Phase 1: LSTM parity baseline (static embeddings, no extra dense layer)
  - Phase 2: Freeze-thaw optimization (trainable embeddings, additional dense layer)
  - Modular architecture supporting both baseline and optimized training modes
  
- **`LSTM.py`**: Straightforward baseline implementation
  - Single-phase training with bidirectional LSTM
  - Comprehensive grid search over units and dropout
  - Static (non-trainable) embeddings throughout

---

## Requirements & Setup

### Dependencies

The project requires Python 3.8+ and the following main packages:

- `tensorflow` >= 2.0 (with GPU support recommended)
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Label encoding and metrics
- `tqdm` - Progress bars
- `kagglehub` - Automatic GloVe download

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Setup (Optional but Recommended)

For GPU acceleration:
1. Install CUDA-compatible TensorFlow: `pip install tensorflow[and-cuda]`
2. Ensure CUDA drivers are properly configured
3. The code automatically sets `LD_LIBRARY_PATH` for CUDA libraries

**Note**: GRU.py automatically detects GPU availability and adjusts batch sizes accordingly.

---

## Data & GloVe Embeddings

### Dataset Format

Both CSV files (`train.csv`, `validation.csv`) must contain:
- **`text`**: Raw text data (tweets, sentences, etc.)
- **`label`**: Emotion category (one of 6 classes)

**Emotion Classes** (mapped 0-5):
```python
LABEL_MAP = {
    0: 'sadness', 1: 'joy', 2: 'love',
    3: 'anger', 4: 'fear', 5: 'surprise'
}
```

### Text Preprocessing

Both pipelines apply identical preprocessing:
```python
def clean_text(text):
    text = str(text).lower()           # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text
```

### Tokenization

- **Vocabulary**: `MAX_WORDS = 30000` (top 30K most frequent words)
- **Sequence Length**: `MAX_LEN = 100` (padded/truncated)
- **OOV Handling**: Out-of-vocabulary words mapped to `<OOV>` token
- **Padding**: Post-padding with zeros for batch training

```python
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['clean_text'])
```

### GloVe Embeddings

- **Source**: GloVe Twitter 27B, 100-dimensional vectors
- **Auto-download**: Via `kagglehub.dataset_download()` on first run
- **Embedding Matrix**: Pre-built matrix aligned to tokenizer vocabulary
  - Rows matching GloVe vocabulary: initialized with pretrained vectors
  - OOV rows: remain zero vectors
  - Coverage: Typically ~70-80% of vocabulary found in GloVe

**LSTM**: Embeddings frozen (`trainable=False`) throughout training
**GRU Phase 1**: Embeddings frozen (LSTM parity)
**GRU Phase 2 & Final**: Embeddings trainable (fine-tuned)

---

---

## LSTM Workflow

**Single-phase baseline training with comprehensive grid search**

### Architecture
```
Input → Embedding (static) → SpatialDropout1D → Bidirectional LSTM → Dense(6, softmax)
```

### Training Process

1. **Load & Preprocess**: Clean text, tokenize, encode labels
2. **Build Embedding Matrix**: Load GloVe vectors, align to vocabulary
3. **Grid Search** (50 combinations):
   - Units: [16, 32, 64, 128, 256]
   - Dropout: [0.05, 0.10, 0.15, ..., 0.50]
   - Each combo trains for 20 epochs with callbacks
4. **Callbacks**:
   - `EarlyStopping`: patience=3 on val_loss
   - `ReduceLROnPlateau`: factor=0.5, patience=2
5. **Output**: Sorted results showing best hyperparameters

### Running LSTM

```bash
python LSTM.py
```

Expected output: Grid search results with validation accuracy for all 50 combinations.

---

## GRU Workflow (Two-Phase Pipeline)

**Advanced training strategy comparing baseline and optimized approaches**

### Phase 1: LSTM Parity Baseline

**Purpose**: Fair comparison with LSTM under identical conditions

**Configuration**:
- Static embeddings (`trainable=False`)
- No extra dense layer (simple architecture)
- 20 epochs, batch_size=32
- Grid: 5 units × 10 dropout = 50 combinations

**Architecture**:
```
Input → Embedding (frozen) → SpatialDropout1D → Bidirectional GRU → Dense(6, softmax)
```

### Phase 2: Freeze-Thaw Optimization

**Purpose**: Explore GRU-specific optimizations

**Configuration**:
- Freeze-thaw training (warmup frozen, then trainable embeddings)
- Extra dense layer with dropout
- Larger batch_size=128 (GPU optimization)
- Grid: 2 units × 2 dropout × 4 learning_rates = 16 combinations

**Architecture**:
```
Input → Embedding (trainable) → SpatialDropout1D → Bidirectional GRU → Dense(64, ReLU) → Dropout → Dense(6, softmax)
```

**Freeze-Thaw Process**:
1. **Warmup** (2-5 epochs): Frozen embeddings, learn task structure
2. **Fine-tune** (3-15 epochs): Unfreeze embeddings, adapt to data

### Phase 3: Final Training

**Configuration**:
- Uses best hyperparameters from Phase 2
- Extended training: 5 warmup + 15 fine-tune epochs
- Full freeze-thaw strategy with EarlyStopping

### Complete Pipeline Flow

```
1. Load Data & Build Embeddings
   ↓
2. Phase 1: Baseline Grid Search (LSTM Parity)
   → 50 combinations, static embeddings
   → Identify best units/dropout
   ↓
3. Phase 2: Optimization Grid Search
   → 16 combinations, freeze-thaw training
   → Identify best learning rates
   ↓
4. Final Training
   → Best Phase 2 params
   → Extended epochs for convergence
   ↓
5. Evaluation & Summary
   → Compare Phase 1 vs Phase 2 vs Final
```

### Running GRU

```bash
python GRU.py
```

Expected output:
- Phase 1 results and best baseline accuracy
- Phase 2 results and best optimized accuracy
- Final model accuracy with comparison summary

---

---

## Hyperparameter Grids

### LSTM Grid (50 combinations)

```python
units_list = [16, 32, 64, 128, 256]
dropout_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# Fixed parameters:
EPOCHS = 20
BATCH_SIZE = 32
spatial_dropout = 0.2 (applied before LSTM)
embeddings: trainable=False
```

### GRU Phase 1 Grid (50 combinations - LSTM Parity)

```python
PHASE1_PARAM_GRID = {
    'gru_units': [16, 32, 64, 128, 256],
    'dropout': [0.05, 0.10, ..., 0.50],  # 10 values
    'spatial_dropout': [0.2],
    'dense_units': [64],  # Not used in Phase 1 architecture
}

# Training config:
EPOCHS = 20
BATCH_SIZE = 32
embeddings: trainable=False
architecture: no extra dense layer
```

### GRU Phase 2 Grid (16 combinations - Optimization)

```python
PHASE2_PARAM_GRID = {
    'gru_units': [128, 256],
    'dropout': [0.2, 0.3],
    'fine_tune_lr': [1e-3, 5e-4, 1e-4, 5e-5],
    'warmup_lr': [1e-3],
    'spatial_dropout': [0.2],
    'dense_units': [64],
}

# Training config:
GRID_WARMUP_EPOCHS = 2
GRID_FINETUNE_EPOCHS = 3
BATCH_SIZE = 128  # Larger for GPU efficiency
embeddings: freeze-thaw (frozen → trainable)
architecture: includes extra Dense(64) + Dropout layer
```

### Final Training Config (GRU)

```python
FINAL_WARMUP_EPOCHS = 5
FINAL_FINETUNE_EPOCHS = 15
BATCH_SIZE = 32
# Uses best parameters from Phase 2
```

### Key Design Decisions

1. **Phase 1 matches LSTM exactly**: Ensures fair comparison
2. **Phase 2 uses larger batches**: Leverages GPU memory efficiently
3. **Grid epochs are short**: Quick exploration (2+3 epochs)
4. **Final training is extended**: Allows full convergence (5+15 epochs)
5. **Learning rates in Phase 2**: Explores fine-tuning sensitivity

---

---

## Examples & Usage

### Basic Usage

```bash
# Run LSTM baseline
python LSTM.py

# Run full GRU two-phase pipeline
python GRU.py
```

### Quick Smoke Test

For rapid validation, reduce epochs in the configuration section:

```python
# In GRU.py, modify:
GRID_WARMUP_EPOCHS = 1
GRID_FINETUNE_EPOCHS = 1
FINAL_WARMUP_EPOCHS = 2
FINAL_FINETUNE_EPOCHS = 3
```

Then run:
```bash
python GRU.py
```

### Custom Data Paths

By default, scripts look for `train.csv` and `validation.csv` in the script directory. To use custom paths:

**Option 1**: Modify constants in the script
```python
# Edit GRU.py or LSTM.py
TRAIN_PATH = "/path/to/your/train.csv"
VAL_PATH = "/path/to/your/validation.csv"
```

**Option 2**: Use absolute paths
```python
BASE_DIR = "/home/user/data"
TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
VAL_PATH = os.path.join(BASE_DIR, "validation.csv")
```

### Understanding Output

**LSTM Output**:
```
LSTM HYPERPARAMETER GRID SEARCH
Training loop with progress bars
FINAL RESULTS SUMMARY
1. LSTM | Units: 128 | Dropout: 0.30 | Val Acc: 0.9234
...
```

**GRU Output**:
```
PHASE 1: BASELINE GRID SEARCH (LSTM Parity)
[Progress for 50 combinations]
✅ PHASE 1 COMPLETE: Best Baseline Accuracy = 0.9156

PHASE 2: FREEZE-THAW OPTIMIZATION GRID SEARCH
[Progress for 16 combinations]
✅ PHASE 2 COMPLETE: Best Optimized Accuracy = 0.9345

FINAL TRAINING: Using Phase 2 best parameters
[Extended training progress]

🎯 FINAL MODEL ACCURACY: 0.9401
   Phase 1 (Baseline): 0.9156
   Phase 2 (Optimized Grid): 0.9345
   Final (Extended Training): 0.9401
```

---

---

## Tips & Troubleshooting

### Common Issues

**GloVe Download Fails**
```bash
# Manual download alternative:
wget https://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
# Update GLOVE_FILE path in the script
```

**GPU Not Detected**
- Scripts automatically fall back to CPU
- Verify CUDA installation: `nvidia-smi`
- Check TensorFlow GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- GRU.py sets `LD_LIBRARY_PATH` automatically for Conda environments

**Memory Issues**
- Reduce `BATCH_SIZE` (especially for Phase 2)
- Reduce `MAX_WORDS` or `MAX_LEN`
- Use smaller GRU/LSTM units in grid search

**Slow Training**
- GPU highly recommended (10-20x speedup)
- Reduce grid search space for experimentation
- Use shorter sequences (`MAX_LEN = 50`)

### Performance Optimization

**For Faster Experimentation**:
1. Reduce grid search space (fewer units/dropout values)
2. Use shorter epochs (1-2 for grid, 5 for final)
3. Smaller `MAX_WORDS` (e.g., 10000)

**For Best Results**:
1. Full grid search (50 combinations for Phase 1)
2. Extended final training (20+ total epochs)
3. GPU with large batch sizes (128+)

### Reproducibility

For deterministic results:
```python
import random
import numpy as np
import tensorflow as tf

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
```

Add this at the beginning of the script before importing models.

---

---

## Developer Notes

### Architecture Decisions

**Why Bidirectional RNNs?**
- Capture context from both past and future
- Significant accuracy improvement for emotion classification
- Standard practice for sequence classification tasks

**Why SpatialDropout1D?**
- More effective than regular dropout for NLP
- Drops entire embedding dimensions rather than individual values
- Reduces overfitting in word embedding layers

**Why Freeze-Thaw Training (GRU)?**
- **Warmup phase**: Learn task-specific patterns without disturbing pretrained knowledge
- **Fine-tune phase**: Adapt embeddings to domain-specific vocabulary
- **Benefit**: Prevents catastrophic forgetting of GloVe semantics
- **Result**: 2-3% accuracy improvement over static embeddings

### Design Principles

1. **Fair Comparison**: Phase 1 GRU matches LSTM exactly (architecture, hyperparameters, training)
2. **Consistency**: All constants (`MAX_LEN`, `MAX_WORDS`, `EMBED_DIM`) identical across models
3. **Modularity**: Separate classes (`GRUModel`, `GRUTrainer`) for reusability
4. **Transparency**: Extensive console output for debugging and monitoring

### Model Comparison

| Aspect | LSTM | GRU Phase 1 | GRU Phase 2 | GRU Final |
|--------|------|-------------|-------------|-----------|
| Embeddings | Frozen | Frozen | Freeze-Thaw | Freeze-Thaw |
| Extra Dense | No | No | Yes | Yes |
| Batch Size | 32 | 32 | 128 | 32 |
| Epochs | 20 | 20 | 2+3 | 5+15 |
| Purpose | Baseline | Fair comparison | Optimization | Best model |

### Extending the Code

**Add New Architectures**:
```python
# In GRUModel.build_model()
# Modify layers list to add CNN, Attention, etc.
```

**Save Results**:
```python
# In GRUTrainer after grid search
import json
with open('phase1_results.json', 'w') as f:
    json.dump(self.grid_results, f, indent=2)
```

**Save Best Model**:
```python
# After final training
final_model.model.save('best_gru_model.keras')
```

**Custom Callbacks**:
```python
# Add to train_freeze_thaw or train_baseline
from tensorflow.keras.callbacks import ModelCheckpoint
callbacks.append(ModelCheckpoint('model_{epoch}.h5', save_best_only=True))
```


### Future Improvements

- [ ] Add test set evaluation
- [ ] Implement classification reports (per-class metrics)
- [ ] Add model saving/loading utilities
- [ ] Support for cross-validation
- [ ] Hyperparameter tuning with Optuna/Hyperband
- [ ] Multi-GPU training support
- [ ] Attention mechanisms
- [ ] Ensemble predictions

---

## License

This project is for educational purposes. GloVe embeddings are subject to their original license terms.

