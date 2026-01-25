# Emotion Classification with BERT-based Models

PyTorch pipeline for fine-tuning Transformer models on 6-class emotion recognition with model compression.

## Project Overview

This project implements **three distinct Transformer architectures** for emotion classification, with support for **model compression** (pruning + quantization) and evaluation metrics designed for **imbalanced datasets**.

**Performance Target:** All models are tuned to achieve **≥80% accuracy** on the validation set.

---

## Models (Triple Architecture Comparison)

We selected three distinct architectures to balance domain specificity with modern efficiency:

| Model | Architecture | HuggingFace ID | Justification |
|-------|--------------|----------------|---------------|
| BERTweet | BERT-based | `vinai/bertweet-base` | Pre-trained on 850M English Tweets; optimized for social media noise (hashtags, slang). |
| CardiffRoBERTa | RoBERTa-based | `cardiffnlp/twitter-roberta-base` | Trained on 58M tweets; proven robust benchmark for sentiment analysis tasks. |
| ModernBERT | BERT-based (efficient) | `answerdotai/ModernBERT-base` | State-of-the-art efficiency (Flash Attention); represents the "next-gen" efficient baseline. |

## Data Preprocessing Pipeline

Raw tweet data is noisy and unstructured. We apply a rigorous cleaning pipeline (`DataCleaner` class) to standardize inputs before tokenization:

1.  **Sanitization:** Removal of URLs, email addresses, and user mentions (`@user`) to prevent overfitting on specific entities.
2.  **Hashtag Handling:** `#hashtag` is converted to `hashtag` to preserve semantic meaning while removing formatting noise.
3.  **Normalization:** Text is lowercased and ASCII-normalized to reduce vocabulary size.
4.  **Filtering:** Duplicate entries and outliers (text length < 3 or > 512) are removed to ensure high-quality training signals.

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

## Usage

### 1. Training

```bash
# Train all 3 models
python emotion_classifier.py --train train.csv --val validation.csv --output ./outputs

# Train specific model
python emotion_classifier.py --model ModernBERT --output ./outputs
```

### 2. Inference 

**Function Signature:**
```python
run_inference(weights: str, csv: str) → list[int]
```

**Inputs:**
- `weights`: Path to trained `.pt` checkpoint file (metadata-rich format)
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

### 3. Model Compression (Tri-Stage)

We implement advanced compression to reduce memory footprint while maintaining accuracy:

| Technique | Method | Description |
|-----------|--------|-------------|
| **Pruning** | L1 Unstructured | Removes 30% of smallest magnitude weights |
| **Quantization** | INT8 (LLM.int8()) | 8-bit precision with outlier protection (FP16 fallback) |
| **Quantization** | NF4 (4-bit) | Normalized Float 4 (optimized for weights) |

**CLI Usage (Benchmarks):**
```bash
# Run All Benchmarks (Original, Pruned, INT8, NF4, Combined)
python emotion_classifier.py --mode compress --weights outputs/best_ModernBERT.pt --val validation.csv
```

**How Compression Works:**
We use a robust pipeline to ensure accuracy preservation:
1.  **Load Fine-Tuned Weights:** Starts with the best-performing FP32 checkpoint.
2.  **Pruning:** Applies 30% unstructured pruning to all linear layers.
3.  **Quantization (NF4/INT8):** Reloads the pruned model using `BitsAndBytesConfig`:
    -   `load_in_4bit=True`: Reduces weight memory by 4x.
    -   `bnb_4bit_quant_type="nf4"`: Optimal data type for normal distributions.
    -   `llm_int8_skip_modules=["classifier"]`: (For INT8) Keeps classification head in FP32 to prevent accuracy collapse.

**Compression Ranking Output:**
The script generates `outputs/compression_results/compression_report.json` comparing:
- **Size (MB):** Model footprint on disk/memory.
- **Inference Latency (ms):** Average time per sample.
- **Macro F1:** Accuracy retention.

---

## Evaluation Metrics (Imbalanced Data)

Due to **class imbalance** in the dataset (e.g., "Joy" is 10x more frequent than "Surprise"), we employ a dual-strategy approach:

### Training Strategy (Active)
We use **Weighted Cross-Entropy Loss** to actively combat imbalance during optimization. Class weights are calculated as the inverse of class frequency:
$$ W_c = \frac{N_{total}}{N_c} $$
This forces the model to treat errors on rare classes (Surprise, Love) as significantly more costly.

### Primary Metrics (Passive Selection)
- **Macro F1 Score** - Treats all classes equally. This is our **primary selection metric**; a model is only considered "best" if it performs well across *all* emotions.
- **Confusion Matrix** - Visual per-class analysis to diagnose specific misclassifications.

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

## Experimental Results

### Baseline Model Performance

We trained all three models on the training set and evaluated them on the validation set (2000 samples). `CardiffRoBERTa` achieved the best performance, slightly outperforming `BERTweet` and `ModernBERT` in Macro F1.

| Model | Accuracy | Macro F1 | Size (MB) | Inference (ms/sample) |
|-------|----------|----------|-----------|-----------------------|
| BERTweet | 0.9395 | 0.9172 | 514.70 | 0.16 |
| **CardiffRoBERTa** | **0.9395** | **0.9178** | **475.58** | **0.16** |
| ModernBERT | 0.9320 | 0.9083 | 570.77 | 0.56 |

*Note: CardiffRoBERTa was selected as the Best Model for compression analysis.*

### Compression Analysis (CardiffRoBERTa)

We applied three compression strategies to the best model. **NF4 (4-bit) quantization** achieved the best trade-off, reducing size by **75.4%** while maintaining **99.4%** of the original Macro F1 score.

| Variant | Accuracy | Macro F1 | Size (MB) | Latency (ms) | Size Reduction |
|---------|----------|----------|-----------|--------------|----------------|
| **Original** | 0.9395 | 0.9178 | 475.58 | 0.24 | 0% |
| **Pruned 30%** | 0.9300 | 0.9092 | 475.58 | 0.16 | 0%* |
| **Quantized INT8** | 0.9385 | 0.9158 | 157.19 | 1.63 | 66.9% |
| **Quantized NF4** | 0.9355 | 0.9127 | 116.99 | 0.65 | **75.4%** |
| **Combined (Pruning + NF4)** | 0.9310 | 0.9104 | 116.99 | 0.64 | 75.4% |

*Key Findings:*
1.  **Size:** NF4 compression is extremely effective, shrinking the model from 476MB to 117MB.
2.  **Accuracy:** Minimal loss (<1% F1 drop) across all methods, validating the robustness of the 3-step pipeline (Load -> Prune -> Quantize).
3.  **Speed:** 4-bit loading (NF4) is faster than 8-bit (INT8) but still slower than native FP16 on small batches due to dequantization overhead. Pruning provided a theoretical speedup but requires sparse kernels to be fully realized.

*\*Note: Unstructured pruning does not reduce file size in standard PyTorch checkpoint format.*

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
    ├── BERTweet_pr_curves.png          # Precision-Recall Curves (New)
    └── class_distribution.png          # Dataset balance check
```

### After Compression
```
compression_results/
├── compression_report.json
└── visualizations/
    └── compression_comparison.png
```

---

## References

1.  **BERTweet:** Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). *BERTweet: A pre-trained language model for English Tweets*. EMNLP.
2.  **Twitter-RoBERTa:** Barbieri, F., Camacho-Collados, J., et al. (2020). *Tweeteval: Unified benchmark and comparative evaluation for tweet classification*. EMNLP.
3.  **ModernBERT:** Answer.AI. (2024). *ModernBERT: A Modern BERT Architecture*.
4.  **LLM.int8():** Dettmers, T., Lewis, M., et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS.
