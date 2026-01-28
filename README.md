# NLP Emotion Classification Projects

This repository contains two distinct projects focusing on **Emotion Classification** from text data (tweets/sentences). The projects explore different generations of NLP architectures, ranging from Recurrent Neural Networks (RNNs) with static embeddings to modern Transformer-based models with compression techniques.

Both projects use the same 6-class emotion dataset:
- **Labels**: 0: Sadness, 1: Joy, 2: Love, 3: Anger, 4: Fear, 5: Surprise
- **Input**: Raw text (tweets)

---

## [Project 1: RNN Baseline & Optimization](./Project1/README.md)

**Directory:** `Project1/`

This project implements and compares Recurrent Neural Network architectures using **GloVe Twitter embeddings** (100d). It focuses on architectural baselines and training strategies.

### Key Features
- **Models**:
  - **LSTM**: Bidirectional LSTM with spatial dropout (Baseline).
  - **GRU**: Bidirectional GRU with a two-phase training pipeline.
- **Techniques**:
  - **Freeze-Thaw Optimization**: A training strategy that warms up the model with frozen embeddings before fine-tuning them to prevent catastrophic forgetting.
  - **Grid Search**: Extensive hyperparameter tuning for units, dropout, and learning rates.
- **Files**:
  - `LSTM.py`: Baseline implementation.
  - `GRU.py`: Advanced two-phase implementation.

[**> Go to Project 1 README**](./Project1/README.md)

---

## [Project 2: Transformers & Model Compression](./Project2/README.md)

**Directory:** `Project2/`

This project leverages state-of-the-art **Transformer models** (BERT-based) for the same classification task. It emphasizes model comparison and efficiency through compression techniques.

### Key Features
- **Models**:
  - **BERTweet**: `vinai/bertweet-base` (Specialized for Tweets).
  - **CardiffRoBERTa**: `cardiffnlp/twitter-roberta-base` (Sentiment baseline).
  - **ModernBERT**: `answerdotai/ModernBERT-base` (High-efficiency architecture).
- **Techniques**:
  - **Fine-tuning**: Transfer learning from pre-trained language models.
  - **Model Compression**:
    - **Pruning**: L1 Unstructured (30% sparsity).
    - **Quantization**: INT8 and NF4 (Normalized Float 4) quantization.
- **Files**:
  - `emotion_classifier.py`: Unified CLI for training, inference, evaluation, and compression.

[**> Go to Project 2 README**](./Project2/README.md)

---

## Setup & Requirements

Each project maintains its own `requirements.txt`. Please refer to the specific project directories for installation instructions.

- **Project 1**: Requires `tensorflow`, `kagglehub`, `pandas`, `numpy`, `scikit-learn`.
- **Project 2**: Requires `torch`, `transformers`, `accelerate`, `bitsandbytes`, `optimum`.
