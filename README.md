# Emotion Classification with RNNs and Transformers

A set of NLP projects for classifying emotions in text using both classical RNN-based models and modern transformer architectures.

The repository explores model design, fine-tuning, and compression techniques across multiple architectures, including LSTM, GRU, and BERT-based models.

**Tech:** Python, PyTorch, TensorFlow, HuggingFace Transformers

--- 

## What Is Implemented
- Bidirectional LSTM and GRU models with GloVe embeddings
- Transformer-based classifiers using BERT variants
- Fine-tuning pipelines for pre-trained models
- Hyperparameter search and structured experiments
- Model compression using pruning and quantization

--- 

Both projects use the same 6-class emotion dataset:
- **Labels**: 0: Sadness, 1: Joy, 2: Love, 3: Anger, 4: Fear, 5: Surprise
- **Input**: Raw text (tweets)

---

## [Project 1: RNN Baseline & Optimization](./Project1/README.md)

**Directory:** `Project1/`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Roeyx/NLP/blob/main/Project1/project1_workflow.ipynb)

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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Roeyx/NLP/blob/main/Project2/emotion_classifier.ipynb)

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
