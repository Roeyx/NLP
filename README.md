# Emotion Detection Project (LSTM & GRU)

This repository implements an emotion classification pipeline using pretrained GloVe embeddings and recurrent neural networks (LSTM and GRU). The main features include:

- Tokenization using Keras `Tokenizer`
- GloVe embedding matrix construction aligned to the tokenizer
- Freeze-thaw training pattern (warmup with frozen embeddings, then fine-tune)
- Hyperparameter grid search (lightweight exploratory grid), followed by a longer final training run using the best parameters
- GPU-aware batch sizing and simple performance-friendly defaults

This README documents the workflow, files, hyperparameters, and recommended usage.

---

## Table of Contents
- Project Structure
- Requirements & Setup
- Data & GloVe Embeddings
- Workflow & Pipeline
- Grid Search & Hyperparameter Design
- Final Training & Evaluation
- Examples: Quick runs & Smoke tests
- Tips & Troubleshooting
- Developer Notes

---

## Project Structure

- `GRU.py` — GRU training pipeline. Modular code with `GRUModel` and `GRUTrainer` classes, supports fine tuning, grid search, and final training, allows usage of GPU.
- `LSTM.py` — LSTM training pipeline. Similar structure to `GRU.py` with tokenization, embedding building and grid searching.
- `data/` — default location for `train.csv` and `validation.csv`.

---

## Requirements & Setup

Install dependencies and set up a Python environment. The repo expects Python 3.8+ and TensorFlow 2.x.

Example setup using pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If running on GPU, install `tensorflow` with GPU support and ensure CUDA drivers are configured.

---

## Data & GloVe Embeddings

- Data format: `train.csv` and `validation.csv` should each contain at least the columns: `text` and `label` (labels are class names or integers depending on your version). Use the `data/` folder or repo root as default locations.
- Tokenization: `Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")` is used; `MAX_WORDS` controls vocabulary size.
- GloVe: We use `glove.twitter.27B.100d.txt` (100 dimension) to build embedding matrices. GRU and LSTM scripts can download the GloVe file via `kagglehub` if not present locally.
- Embedding matrix: aligned to `tokenizer.word_index`. OOV rows are left zeros (same as our LSTM approach).

---

## Workflow & Pipeline (Detailed)

1. **Load & Clean Data**
   - Load CSVs with pandas, apply `clean_text()` to lowercase & remove non-alpha characters.
2. **Tokenize & Pad**
   - Fit a Keras `Tokenizer` on the training `clean_text`, convert train/val texts to sequences and pad using `pad_sequences` with `MAX_LEN`.
3. **Label Encoding**
   - Use `LabelEncoder` from scikit-learn to convert text labels to integers.
4. **Build Embedding Matrix**
   - Read the GloVe file, build a dense `embedding_matrix` aligned to tokenizer indices. Leave zeros for OOV tokens.
5. **Grid Search (short runs)**
   - The `GRUTrainer.run_grid_search()` builds models per parameter combination, runs a short freeze-thaw training, evaluates on validation, and stores results in-memory.
   - It prints per-run validation scores and a final sorted summary.
6. **Final Run**
   - Rebuild the model with `best_params` and run `train_freeze_thaw` with the longer `FINAL_WARMUP_EPOCHS` and `FINAL_FINETUNE_EPOCHS`. The script prints the final validation accuracy.

---

## Grid Search & Hyperparameter Design

We use a small but informative grid to keep experiments practical:

- GRU Example (current default):
  - `gru_units`: [128, 256]
  - `dropout`: [0.2, 0.3]
  - `fine_tune_lr`: [1e-3, 5e-4, 1e-4, 5e-5]
  - `warmup_lr`: [1e-3] (kept fixed in the grid)
  - `spatial_dropout`: [0.2]
  - `dense_units`: [64]

This yields 2 × 2 × 4 = 16 combinations (i.e. manageable). Adjust as needed.

Grid runs are purposely short (GRID_WARMUP_EPOCHS & GRID_FINETUNE_EPOCHS small) to quickly identify promising configurations. The final training uses longer epoch counts.

---

## Final Training & Evaluation

- Final training uses the best parameters from grid search and trains for `FINAL_WARMUP_EPOCHS` + `FINAL_FINETUNE_EPOCHS`.
- The embedding layer is fine-tuned (`trainable=True`) in the final training.
- Final evaluation prints the final validation accuracy; any classification report logging is removed by default to keep console output compact.

---

## Examples: Quick Runs & Smoke Tests

Minimal quick smoke test — reduce grid and epoch counts for fast validation:

```bash
# Edit GRU.py to set GRID_WARMUP_EPOCHS = 1, GRID_FINETUNE_EPOCHS = 1
python3 GRU.py
```

Full run:

```bash
python3 GRU.py
```

Set `TRAIN_PATH`, `VAL_PATH`, and `GLOVE_FILE` environment variables or edit `GRU.py` to point to the data if they're located elsewhere.

---

## Tips & Troubleshooting

- If `kagglehub` fails to download GloVe, manually place `glove.twitter.27B.100d.txt` in the repo root or `data/` and set `GLOVE_FILE` accordingly.
- If TensorFlow can't access GPU, the script will detect and fall back to CPU (batch size = 32).
- For reproducibility, set seeds in both Python and TensorFlow. Use deterministic settings if needed.
- Keep the grid small when iterating. Use short runs (1–3 epochs) during exploration, then a final long run (e.g., `FINAL_FINETUNE_EPOCHS = 15`) when you know the best params.

---

## Developer Notes

- Freezing embeddings during warmup stabilizes early training and reduces catastrophic forgetting; unfreezing for fine-tuning improves final accuracy.
- Keep `MAX_LEN`, `MAX_WORDS`, and `EMBED_DIM` consistent across LSTM & GRU runs for valid comparisons.
- We intentionally keep outputting results to the console for clarity, but you can add optional saving of `grid_results.json` and final model saving if you want to persist results.

