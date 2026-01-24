# Emotion Classification with BERT-based Models

PyTorch pipeline for fine-tuning Transformer models on 6-class emotion recognition with model compression.

## Project Overview

This project implements **three distinct Transformer architectures** for emotion classification, with support for **model compression** (pruning + quantization) and evaluation metrics designed for **imbalanced datasets**.

**Performance Target:** All models are tuned to achieve **≥80% accuracy** on the validation set.

---

## Models (Triple Architecture Comparison)

| Model | Architecture | HuggingFace ID |
|-------|--------------|----------------|
| BERTweet | BERT-based | `vinai/bertweet-base` |
| CardiffRoBERTa | RoBERTa-based | `cardiffnlp/twitter-roberta-base` |
| ModernBERT | BERT-based (efficient) | `answerdotai/ModernBERT-base` |

## Class Mapping (6 Emotions)

| Label | Emotion |
|-------|---------|
| 0 | Sadness |
| 1 | Joy |
| 2 | Love |
| 3 | Anger |
| 4 | Fear |
| 5 | Surprise |

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.8, CUDA GPU (recommended)

---

## Modular Structure

The codebase is organized into three main modules:

| Module | Function | Description |
|--------|----------|-------------|
| **Training** | `main()` | Fine-tunes models, saves checkpoints and metrics |
| **Inference** | `run_inference()` | Loads weights, predicts on new data |
| **Compression** | `compress_and_evaluate()` | Applies pruning/quantization, compares performance |

---

## Usage

### 1. Training

```bash
# Train all 3 models
python emotion_classifier.py --train train.csv --val validation.csv --output ./outputs --gpu 0

# Train specific model
python emotion_classifier.py --model ModernBERT --output ./outputs --gpu 1
```

### 2. Inference (Mandatory Interface)

**Function Signature:**
```python
run_inference(weights: str, csv: str) → list[int]
```

**Inputs:**
- `weights`: Path to trained `.pt` checkpoint file
- `csv`: Path to CSV file with `text` column (**labels NOT required**)

**Outputs:**
- Returns list of predicted labels (0-5)
- Saves `<input_name>_predictions.csv` with predictions

**CLI Usage:**
```bash
python emotion_classifier.py --mode inference --weights outputs/best_ModernBERT.pt --test_csv test.csv
```

**Python Usage:**
```python
from emotion_classifier import run_inference

# Test CSV only needs 'text' column - no labels required
predictions = run_inference('outputs/best_ModernBERT.pt', 'test.csv')
# → Returns: [0, 1, 3, 2, ...] (predicted label integers)
# → Saves: test_predictions.csv
```

### 3. Model Compression (Dual Techniques)

Two compression methods applied to the **best-performing model**:

| Technique | Method | Description |
|-----------|--------|-------------|
| **Pruning** | L1 Unstructured | Removes 30% of smallest magnitude weights |
| **Quantization** | BitsAndBytes INT8 | Quantizes weights to INT8, protects classifier head in FP32 |

**CLI Usage (Separate Benchmarks):**
```bash
# Run Pruning Benchmark Only
python emotion_classifier.py --mode compress --weights outputs/best_ModernBERT.pt --val validation.csv --technique prune

# Run Quantization Benchmark Only
python emotion_classifier.py --mode compress --weights outputs/best_ModernBERT.pt --val validation.csv --technique quantize

# Run Combined (Pruning + Quantization)
python emotion_classifier.py --mode compress --weights outputs/best_ModernBERT.pt --val validation.csv --technique combined

# Run All (Default)
python emotion_classifier.py --mode compress --weights outputs/best_ModernBERT.pt --val validation.csv
```

**How Compression Works:**

We implement a robust 3-stage pipeline to ensure accuracy is preserved:
1.  **Load Fine-Tuned Weights:** Starts with the best-performing FP32 checkpoint.
2.  **Pruning:** Applies 30% unstructured pruning to all linear layers.
3.  **INT8 Quantization:** Reloads the pruned model using `BitsAndBytesConfig` with:
    -   `load_in_8bit=True`: Reduces weight memory by 4x.
    -   `llm_int8_threshold=6.0`: Handles outlier activations in FP16.
    -   `llm_int8_skip_modules=["classifier"]`: **Crucially**, keeps the classification head in FP32 to prevent accuracy collapse.

**Compression Ranking Output:**

The compression report ranks models by size and speed:
```
outputs/compression_results/
├── compression_report.json          # Metrics comparison
└── visualizations/
    └── compression_comparison.png   # Bar charts
```

| Variant | Size | Speed | Accuracy |
|---------|------|-------|----------|
| Original | 100% | 1.0x | Baseline |
| Pruned 30% | ~100%* | ~1.5x | ~Same |
| Quantized INT8 | ~35% | ~0.16x** | ~Same |
| Combined | ~35% | ~0.20x | ~Same |

*\*Note: Unstructured pruning doesn't reduce file size without specialized sparse storage, but improves inference speed on supported hardware.*
*\**Note: BitsAndBytes INT8 quantization reduces memory usage by ~65% but may increase latency on small batches due to kernel overhead.*

---

## Evaluation Metrics (Imbalanced Data)

Due to **class imbalance** in the dataset, we use:

### Primary Metrics
- **Macro F1 Score** - Treats all classes equally (used for model selection)
- **Confusion Matrix** - Visual per-class analysis

### Secondary Metrics
- **Accuracy** - Overall correctness (must exceed 80%)
- **Inference Time** - Latency in ms/sample
- **Model Size** - Memory footprint in MB
- **Per-class Precision/Recall/F1** - Detailed breakdown

All metrics are logged to console and saved to `emotion_classification_report.json`.

---

## Outputs & Deliverables

### After Training
```
outputs/
├── best_BERTweet.pt                    # Checkpoint (portable format)
├── best_CardiffRoBERTa.pt
├── best_ModernBERT.pt
├── emotion_classification_report.json  # All metrics for article
└── visualizations/
    ├── BERTweet_confusion_matrix.png
    ├── BERTweet_training_curves.png
    ├── CardiffRoBERTa_confusion_matrix.png
    ├── CardiffRoBERTa_training_curves.png
    ├── ModernBERT_confusion_matrix.png
    └── ModernBERT_training_curves.png
```

### After Compression
```
compression_results/
├── compression_report.json
└── visualizations/
    └── compression_comparison.png
```

### After Inference
```
test_predictions.csv    # Predictions for submission
```

---

## Report Data Generation

All tables and plots for the academic article are **automatically generated**:

| Deliverable | Generated File |
|-------------|----------------|
| Model comparison table | `emotion_classification_report.json` |
| Confusion matrices | `visualizations/*_confusion_matrix.png` |
| Training curves | `visualizations/*_training_curves.png` |
| Compression comparison | `compression_report.json` + `compression_comparison.png` |
| Per-class metrics | Included in JSON reports |

---

## Reproducibility

### One-Step Full Pipeline
```bash
# 1. Train all models
python emotion_classifier.py --train train.csv --val validation.csv --output ./outputs --gpu 0

# 2. Run compression on best model
python emotion_classifier.py --mode compress --weights outputs/best_ModernBERT.pt --val validation.csv --output ./outputs

# 3. Generate predictions for test set
python emotion_classifier.py --mode inference --weights outputs/best_ModernBERT.pt --test_csv test.csv
```

### Reproducibility Features
- **Fixed random seed** (42) with deterministic CUDA operations
- **Checkpoints include metadata** (model_id, config, hyperparameters)
- **Backward-compatible checkpoint loading**

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 32 | Reduce to 16 if OOM |
| `MAX_LENGTH` | 128 | Token sequence length |
| `EPOCHS` | 5 | Training epochs |
| `LEARNING_RATE` | 5e-5 | AdamW learning rate |
| `SEED` | 42 | Random seed for reproducibility |

---

## Python API

```python
from emotion_classifier import main, run_inference, compress_and_evaluate

# Training
main(train_path='train.csv', val_path='validation.csv', output_dir='./outputs')

# Inference (works without labels in CSV)
predictions = run_inference('outputs/best_ModernBERT.pt', 'test.csv')

# Compression analysis
results = compress_and_evaluate('outputs/best_ModernBERT.pt', 'validation.csv')
```

---

## Weight File Format

Checkpoints are saved in portable PyTorch format (`.pt`) with embedded metadata:

```python
{
    'model_state_dict': ...,      # Model weights
    'model_id': 'vinai/bertweet-base',
    'model_name': 'BERTweet',
    'config': {
        'max_length': 128,
        'num_labels': 6,
        ...
    }
}
```

This ensures weights can be loaded without external configuration files.
