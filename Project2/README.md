# Emotion Classification with BERT-based Models

This repository contains the implementation for the paper "Emotion Classification with BERT-based Models". It provides a pipeline for fine-tuning Transformer models (BERTweet, CardiffRoBERTa, ModernBERT) on 6-class emotion recognition, along with compression techniques (Pruning + Quantization).

## 🚀 Quick Start (For Graders/Inference)

To run inference on a new test dataset using our pre-trained weights:

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Ensure your input CSV has a column named **`text`** containing the tweets/sentences. Labels are not required.
*Example `test.csv`:*
```csv
id,text
1,"I am feeling so happy today!"
2,"This is actually quite terrifying."
```

### 3. Run Inference
```bash
# General Syntax
python emotion_classifier.py --mode inference --weights <path_to_checkpoint> --test_csv <path_to_csv>

# Example (using provided ModernBERT weights)
python emotion_classifier.py --mode inference --weights outputs/best_ModernBERT.pt --test_csv test.csv
```

**Output:**
- The script generates a prediction file in the same directory as the input CSV (e.g., `test_predictions.csv`).
- The output CSV includes the original data plus `predicted_label` (0-5) and `predicted_emotion` (String).

---

## Project Overview

### Task
Classify text into 6 basic emotions:
| Label | Emotion |
|-------|---------|
| 0 | Sadness |
| 1 | Joy |
| 2 | Love |
| 3 | Anger |
| 4 | Fear |
| 5 | Surprise |

### Models
We compare three architectures:
1.  **BERTweet** (`vinai/bertweet-base`): Specialized for Tweets.
2.  **CardiffRoBERTa** (`cardiffnlp/twitter-roberta-base`): Robust sentiment baseline.
3.  **ModernBERT** (`answerdotai/ModernBERT-base`): High-efficiency modern architecture.

---

## Reproducing Paper Results

### 1. Training
To retrain the models from scratch:

```bash
# Train all models (BERTweet, CardiffRoBERTa, ModernBERT)
python emotion_classifier.py --train train.csv --val validation.csv --output ./outputs

# Train a specific model only
python emotion_classifier.py --train train.csv --val validation.csv --model ModernBERT --output ./outputs
```
*Note: Training includes automatic evaluation on the validation set and saves the checkpoint with the best Macro F1 score.*

### 2. Evaluation
To evaluate a trained model on labeled data (calculates Accuracy, Macro F1, and Confusion Matrix):

```bash
python emotion_classifier.py --mode evaluate --weights outputs/best_CardiffRoBERTa.pt --val validation.csv
```

### 3. Compression Benchmarks
To reproduce the pruning and quantization results reported in the paper:

```bash
# Run full compression suite (Pruning, INT8, NF4, Combined)
python emotion_classifier.py --mode compress --weights outputs/best_ModernBERT.pt --val validation.csv --technique all
```

**Techniques Evaluated:**
*   **Pruning:** L1 Unstructured (30% sparsity).
*   **Quantization (INT8):** `LLM.int8()` mixed-precision decomposition.
*   **Quantization (NF4):** Normalized Float 4 (via `bitsandbytes`).

---

## Directory Structure

```
.
├── emotion_classifier.py    # Main entry point for Training, Inference, and Compression
├── requirements.txt         # Dependencies
├── train.csv                # (Required for training)
├── validation.csv           # (Required for training/eval)
├── outputs/                 # Generated checkpoints and logs
│   ├── best_ModernBERT.pt
│   ├── emotion_classification_report.json
│   └── visualizations/      # Confusion matrices, Loss curves, PR curves
└── AGENTS.md                # Agent guidelines
```

## Troubleshooting

**OOM (Out of Memory) Errors:**
If running on a GPU with <8GB VRAM, edit `emotion_classifier.py` and reduce `BATCH_SIZE` from 32 to 16.

**Model ID Errors:**
The checkpoints save metadata including the HuggingFace model ID. If you encounter an error regarding the model config, you can force a specific architecture using the `--model_id` flag:
```bash
python emotion_classifier.py --mode inference --weights old_checkpoint.pt --test_csv test.csv --model_id vinai/bertweet-base
```
