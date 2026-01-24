#!/usr/bin/env python3
"""
Emotion Classification with BERT-based Models
Models: BERTweet, CardiffRoBERTa, ModernBERT
Task: 6-class emotion classification

Usage:
    Training (all models):
        python emotion_classifier.py --train train.csv --val validation.csv --output ./outputs

    Training (single model):
        python emotion_classifier.py --model BERTweet --output ./outputs

    Inference on test set:
        python emotion_classifier.py --mode inference --weights best_ModernBERT.pt --test_csv test.csv

    Compression analysis:
        python emotion_classifier.py --mode compress --weights best_ModernBERT.pt --val validation.csv

    In-code:
        from emotion_classifier import main, run_inference, compress_and_evaluate
        main(train_path='train.csv', val_path='validation.csv', output_dir='./outputs')
        predictions = run_inference('best_ModernBERT.pt', 'test.csv')

Requirements:
    - Python >= 3.8
    - See requirements.txt for dependencies
    - GPU recommended (CUDA), CPU fallback supported
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from tqdm.auto import tqdm
import re
from collections import Counter
import json
import gc
import os
import argparse

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== CONFIGURATION ====================
# Default device - can be overridden via CLI --gpu argument
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_device(gpu_id: int = None):
    """
    Set the compute device.
    
    Args:
        gpu_id: GPU device ID (0, 1, etc.) or None for auto-detect
    
    Returns:
        torch.device
    """
    global DEVICE
    if gpu_id is not None and torch.cuda.is_available():
        if gpu_id < torch.cuda.device_count():
            DEVICE = torch.device(f'cuda:{gpu_id}')
            torch.cuda.set_device(gpu_id)
        else:
            print(f"⚠️ GPU {gpu_id} not available. Using GPU 0.")
            DEVICE = torch.device('cuda:0')
    elif torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')
    return DEVICE


EMOTION_LABELS = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

MODELS = {
    'BERTweet': 'vinai/bertweet-base',
    'CardiffRoBERTa': 'cardiffnlp/twitter-roberta-base',
    'ModernBERT': 'answerdotai/ModernBERT-base'
}

# Hyperparameters
BATCH_SIZE = 32  # Reduce to 16 if OOM
EPOCHS = 5
LEARNING_RATE = 5e-5
MAX_LENGTH = 128
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01

# INT8 Quantization uses BitsAndBytes LLM.int8() with outlier handling
# Threshold of 6.0 keeps outlier activations in FP16 for accuracy preservation


def _print_config(gpu_id=None):
    """Print configuration info. Called only when running as script."""
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        device_idx = gpu_id if gpu_id is not None else 0
        if device_idx < torch.cuda.device_count():
            print(f"GPU {device_idx}: {torch.cuda.get_device_name(device_idx)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    print("\n⚙️ Configuration:")
    print(f" Batch Size: {BATCH_SIZE}")
    print(f" Max Length: {MAX_LENGTH}")
    print(f" Epochs: {EPOCHS}")
    print(f" Learning Rate: {LEARNING_RATE}")
    print(f" Warmup Steps: {WARMUP_STEPS}")
    print(f" Weight Decay: {WEIGHT_DECAY}")


# ==================== DATA CLEANER ====================
class DataCleaner:
    """Utilities to sanitize and filter raw text data."""
    
    @staticmethod
    def clean_text(text):
        """Remove URLs, emails, mentions, hashtags, non-ascii chars."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    @staticmethod
    def remove_duplicates(df):
        """Drop duplicate text entries."""
        initial = len(df)
        df = df.drop_duplicates(subset=['text'], keep='first')
        print(f" Removed {initial - len(df)} duplicates")
        return df

    @staticmethod
    def remove_outliers(df, min_len=3, max_len=512):
        """Filter by text length."""
        initial = len(df)
        df = df[(df['text'].str.len() >= min_len) & (df['text'].str.len() <= max_len)]
        print(f" Removed {initial - len(df)} outliers")
        return df

    @staticmethod
    def handle_missing_values(df):
        """Remove rows missing text or label."""
        return df.dropna(subset=['text', 'label'])


# ==================== EMOTION DATASET ====================
class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion classification with labels."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx]
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class InferenceDataset(Dataset):
    """PyTorch Dataset for inference without labels (external test sets)."""
    
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }


# ==================== EMOTION CLASSIFIER ====================
class EmotionClassifier:
    """
    Encapsulates a HuggingFace transformer for sequence classification.
    
    Features:
    - Loads tokenizer and model with num_labels=6
    - Supports class weights for imbalanced data
    - Saves checkpoints with metadata for reproducibility
    - Generates training curves visualization
    """
    
    def __init__(self, model_name, model_id, device, class_weights=None):
        self.model_name = model_name
        self.model_id = model_id
        self.device = device
        self.class_weights = class_weights

        print(f"\nLoading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=6,
            ignore_mismatched_sizes=True
        )

        # Full fine-tuning: all params trainable
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f" 🔓 Full Fine-tuning: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")

        self.model.to(device)
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.best_val_loss = float('inf')
        self.train_time = 0

    def get_loss_fn(self):
        """Get loss function with optional class weights."""
        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, dtype=torch.float).to(self.device)
            return nn.CrossEntropyLoss(weight=weights)
        return nn.CrossEntropyLoss()

    def train_epoch(self, train_loader, optimizer, scheduler, loss_fn):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training {self.model_name}", leave=False):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate(self, val_loader, loss_fn):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Eval {self.model_name}", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        return total_loss / len(val_loader), acc, all_preds, all_labels

    def train(self, train_loader, val_loader, num_epochs=5, output_dir='./outputs'):
        """
        Full training loop with early stopping and checkpoint saving.
        
        Saves checkpoint with metadata for reproducibility.
        """
        loss_fn = self.get_loss_fn()
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=total_steps
        )

        patience = 3
        patience_counter = 0
        start_time = time.time()
        ckpt_path = os.path.join(output_dir, f'best_{self.model_name}.pt')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs} - {self.model_name}")
            epoch_start = time.time()
            train_loss = self.train_epoch(train_loader, optimizer, scheduler, loss_fn)
            val_loss, val_acc, _, _ = self.evaluate(val_loader, loss_fn)
            epoch_time = time.time() - epoch_start

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.1f}s")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                # Save checkpoint with metadata (.pt format for portability)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'model_id': self.model_id,
                    'model_name': self.model_name,
                    'config': {
                        'max_length': MAX_LENGTH,
                        'num_labels': 6,
                        'batch_size': BATCH_SIZE,
                        'learning_rate': LEARNING_RATE,
                        'weight_decay': WEIGHT_DECAY,
                    },
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'epoch': epoch + 1,
                }, ckpt_path)
                
                # Also save in HuggingFace format (for BitsAndBytes quantization compatibility)
                hf_save_dir = ckpt_path.replace('.pt', '')  # e.g., outputs/best_BERTweet
                self.model.save_pretrained(hf_save_dir)
                self.tokenizer.save_pretrained(hf_save_dir)
                
                print(f" Saved checkpoint: {ckpt_path} (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f" Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping (patience={patience})")
                    break

        self.train_time = time.time() - start_time
        
        # Load best checkpoint with backward compatibility
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

    def save_training_curves(self, output_dir):
        """Save training loss and validation accuracy curves."""
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        axes[0].plot(epochs, self.history['train_loss'], 'b-o', label='Train Loss', markersize=4)
        axes[0].plot(epochs, self.history['val_loss'], 'r-o', label='Val Loss', markersize=4)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{self.model_name} - Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[1].plot(epochs, self.history['val_acc'], 'g-o', label='Val Accuracy', markersize=4)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'{self.model_name} - Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        curve_path = os.path.join(vis_dir, f'{self.model_name}_training_curves.png')
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" 📈 Training curves saved: {curve_path}")

    def predict(self, test_loader):
        """Run prediction and return metrics."""
        print(f"\nPredicting on {len(test_loader.dataset)} samples using {self.model_name}...")
        self.model.eval()
        all_preds, all_labels, all_logits = [], [], []
        inference_times = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Predict {self.model_name}", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                start = time.time()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                elapsed = time.time() - start
                inference_times.append(elapsed / len(labels))

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
                
        avg_time = np.mean(inference_times) if inference_times else 0
        print(f"Prediction complete. Avg inference: {avg_time*1000:.2f} ms/sample")
        return all_preds, all_labels, all_logits, avg_time


# ==================== DATA LOADING HELPERS ====================
def load_and_prepare_data(train_path='train.csv', val_path='validation.csv'):
    """Load and preprocess CSV data."""
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
    except Exception as e:
        print("Error: CSV files not found!", e)
        return None, None

    print(f"Loaded: {train_path} -> {len(train_df)} rows")
    print(f"Loaded: {val_path} -> {len(val_df)} rows")
    
    cleaner = DataCleaner()
    train_df = cleaner.handle_missing_values(train_df)
    val_df = cleaner.handle_missing_values(val_df)
    train_df['text'] = train_df['text'].apply(cleaner.clean_text)
    val_df['text'] = val_df['text'].apply(cleaner.clean_text)
    train_df = cleaner.remove_duplicates(train_df)
    val_df = cleaner.remove_duplicates(val_df)
    train_df = cleaner.remove_outliers(train_df)
    val_df = cleaner.remove_outliers(val_df)

    print(f"Cleaned: {len(train_df)} train, {len(val_df)} val")
    return train_df, val_df


def get_class_weights(labels):
    """Compute class weights for imbalanced data."""
    counts = Counter(labels)
    total = len(labels)
    counts_list = [counts.get(i, 0) for i in range(6)]
    weights = [total / counts[i] for i in range(6)]
    weights = np.array(weights)
    weights = weights / weights.sum() * 6
    print("Class counts:", counts_list)
    print("Class weights:", np.round(weights, 4))
    return weights


def get_model_size_mb(model):
    """Calculate model size in MB by saving to a temp file."""
    import tempfile
    with tempfile.NamedTemporaryFile() as tmp:
        torch.save(model.state_dict(), tmp.name)
        size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
    return size_mb


# ==================== INFERENCE FUNCTION ====================
def run_inference(weights: str, csv: str, model_id: str = None,
                  text_column: str = 'text', output_csv: str = None) -> list:
    """
    Run inference on external test set using trained weights.
    
    Args:
        weights: Path to .pt checkpoint file
        csv: Path to CSV file containing text data
        model_id: HuggingFace model ID (optional if saved in checkpoint)
        text_column: Name of text column in CSV (default: 'text')
        output_csv: Path to save predictions (optional, auto-generated if None)
    
    Returns:
        List of predicted label integers (0-5)
    
    Example:
        predictions = run_inference('best_ModernBERT.pt', 'test.csv')
    """
    print(f"\n{'='*50}")
    print("RUNNING INFERENCE")
    print(f"{'='*50}")
    print(f" Weights: {weights}")
    print(f" CSV: {csv}")
    
    # Load checkpoint with backward compatibility
    checkpoint = torch.load(weights, map_location=DEVICE)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        saved_model_id = checkpoint.get('model_id')
        max_length = checkpoint.get('config', {}).get('max_length', MAX_LENGTH)
        model_id = model_id or saved_model_id
        print(f" Checkpoint format: New (metadata included)")
        print(f" Model ID: {saved_model_id}")
    else:
        state_dict = checkpoint
        max_length = MAX_LENGTH
        print(f" Checkpoint format: Legacy (state_dict only)")
    
    if model_id is None:
        raise ValueError(
            "model_id is required. Either provide it as argument or use new checkpoint format.\n"
            f"Available models: {list(MODELS.keys())}"
        )
    
    print(f" Using model_id: {model_id}")
    print(f" Max length: {max_length}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=6, ignore_mismatched_sizes=True
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    # Load and preprocess CSV
    df = pd.read_csv(csv)
    print(f" Loaded {len(df)} rows")
    df[text_column] = df[text_column].apply(DataCleaner.clean_text)
    
    # Create dataset and dataloader
    dataset = InferenceDataset(df[text_column], tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Run inference
    predictions = []
    inference_times = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            start = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            elapsed = time.time() - start
            inference_times.append(elapsed)
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy().tolist())
    
    avg_time = sum(inference_times) / len(predictions) * 1000
    print(f"\n Inference complete: {len(predictions)} predictions")
    print(f" Avg time: {avg_time:.2f} ms/sample")
    
    # Save predictions to CSV
    if output_csv is None:
        base_name = os.path.splitext(os.path.basename(csv))[0]
        output_csv = f"{base_name}_predictions.csv"
    
    result_df = df.copy()
    result_df['predicted_label'] = predictions
    result_df['predicted_emotion'] = [EMOTION_LABELS[p] for p in predictions]
    result_df.to_csv(output_csv, index=False)
    print(f" Predictions saved: {output_csv}")
    
    return predictions


# ==================== MODEL COMPRESSION ====================
def apply_pruning(model, amount=0.3):
    """
    Apply unstructured L1 pruning to Linear layers.
    
    Args:
        model: PyTorch model
        amount: Fraction of weights to prune (default: 0.3 = 30%)
    
    Returns:
        Pruned model (modified in-place)
    """
    pruned_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # Make permanent
            pruned_count += 1
    
    print(f"  Pruned {pruned_count} Linear layers at {amount*100:.0f}% sparsity")
    return model





def evaluate_compressed_model(model, val_loader, name, device=None, output_dir=None):
    """
    Evaluate a model and return detailed metrics.
    
    Returns dict with accuracy, macro_f1, size_mb, inference_time_ms,
    plus classification report and confusion matrix.
    """
    device = device or DEVICE
    
    # BitsAndBytes 8-bit models cannot be moved manually as they are already device-mapped
    if not getattr(model, "is_loaded_in_8bit", False):
        model.to(device)
        
    model.eval()
    
    all_preds, all_labels = [], []
    total_time = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Eval {name}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            start = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            total_time += time.time() - start
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    size_mb = get_model_size_mb(model)
    avg_time_ms = (total_time / len(all_preds)) * 1000
    
    # Detailed Analysis (Imbalance Handling)
    cr = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save Confusion Matrix if output_dir is provided
    if output_dir:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(EMOTION_LABELS.values()),
                    yticklabels=list(EMOTION_LABELS.values()))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{name} - Confusion Matrix')
        
        # Clean name for filename (remove parentheses, spaces)
        safe_name = re.sub(r'[^\w\-_]', '_', name)
        cm_path = os.path.join(output_dir, 'visualizations', f'{safe_name}_confusion_matrix.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved confusion matrix: {cm_path}")
    
    return {
        'name': name,
        'accuracy': acc,
        'macro_f1': macro_f1,
        'size_mb': size_mb,
        'inference_time_ms': avg_time_ms,
        'num_samples': len(all_preds),
        'per_class_report': cr,
        'confusion_matrix': cm.tolist()
    }


def load_compressed_model(model_dir: str, prune_amount: float = 0.0):
    """
    Load model with INT8 Quantization using BitsAndBytes, optionally with Pruning.
    
    Args:
        model_dir: Path to HF model directory
        prune_amount: Fraction of weights to prune before quantization (0.0 to disable)
    """
    if prune_amount > 0:
        print(f"\n  Creating Compressed Model (Pruning {prune_amount:.0%} + INT8)...")
    else:
        print("\n  Creating Compressed Model (INT8 Only)...")
        
    from transformers import BitsAndBytesConfig
    
    # 1. Load fine-tuned model (FP32)
    print("   1. Loading fine-tuned weights...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=6, ignore_mismatched_sizes=True
    )
    
    # 2. Apply Pruning (Optional)
    if prune_amount > 0:
        print(f"   2. Applying {prune_amount:.0%} Unstructured Pruning...")
        pruned_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=prune_amount)
                prune.remove(module, 'weight')  # Make permanent
                pruned_count += 1
        print(f"      Pruned {pruned_count} layers.")
    
    # Save temp model (standard format for BitsAndBytes reloading)
    temp_dir = "./temp_compressed_model"
    model.save_pretrained(temp_dir)
    del model
    gc.collect()
    
    # 3. Reload with INT8 Quantization
    print(f"   {3 if prune_amount > 0 else 2}. Reloading with BitsAndBytes INT8 (skip_modules=['classifier'])...")
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,          # Handle outliers in FP16
        llm_int8_skip_modules=["classifier"]  # PROTECT THE HEAD
    )
    
    compressed_model = AutoModelForSequenceClassification.from_pretrained(
        temp_dir,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    # Clean up
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        
    return compressed_model


def compress_and_evaluate(weights_path: str, val_csv: str, model_id: str = None,
                          output_dir: str = './compression_results',
                          technique: str = 'all') -> dict:
    """
    Apply compression techniques and compare performance.
    
    Args:
        weights_path: Path to trained model checkpoint (.pt)
        val_csv: Path to validation dataset CSV
        model_id: HuggingFace model ID (optional if in checkpoint)
        output_dir: Directory to save reports and charts
        technique: Compression mode ('prune', 'quantize', 'combined', or 'all')
    
    Returns:
        Dictionary containing metrics for all evaluated models.
    """
    print(f"\n{'='*70}")
    print(f"MODEL COMPRESSION ANALYSIS ({technique.upper()})")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        model_id = model_id or checkpoint.get('model_id')
        max_length = checkpoint.get('config', {}).get('max_length', MAX_LENGTH)
    else:
        state_dict = checkpoint
        max_length = MAX_LENGTH
    
    if model_id is None:
        raise ValueError("model_id required")
    
    # Derive HuggingFace save directory from weights path
    # e.g., outputs/best_BERTweet.pt -> outputs/best_BERTweet
    hf_model_dir = weights_path.replace('.pt', '')
    
    print(f"  Model: {model_id}")
    print(f"  Weights: {weights_path}")
    print(f"  HF Dir: {hf_model_dir}")
    
    # Prepare validation data
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    val_df = pd.read_csv(val_csv)
    val_df['text'] = val_df['text'].apply(DataCleaner.clean_text)
    val_ds = EmotionDataset(val_df['text'], val_df['label'], tokenizer, max_length)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    def load_fresh_model():
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=6, ignore_mismatched_sizes=True
        )
        model.load_state_dict(state_dict)
        return model
    
    results = {}
    
    # 1. Original Model (Baseline)
    # Always run baseline to ensure self-contained report
    print("\n  Evaluating Original Model (GPU)...")
    original = load_fresh_model().to(DEVICE)
    results['original'] = evaluate_compressed_model(original, val_loader, 'Original (GPU)', output_dir=output_dir)
    del original
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # 2. Pruned Model
    if technique in ['prune', 'all']:
        print("\n  Applying 30% Pruning...")
        pruned = load_fresh_model()
        pruned = apply_pruning(pruned, amount=0.3)
        pruned.to(DEVICE)
        results['pruned_30'] = evaluate_compressed_model(pruned, val_loader, 'Pruned 30%', output_dir=output_dir)
        del pruned
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # 3. Quantized Model
    if technique in ['quantize', 'all']:
        # Ensure HF directory exists for BitsAndBytes loading
        if not os.path.isdir(hf_model_dir):
            print(f"  HF directory not found. Converting {weights_path} to HF format at {hf_model_dir}...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = load_fresh_model()
                model.save_pretrained(hf_model_dir)
                tokenizer.save_pretrained(hf_model_dir)
                del model
                print("  Conversion complete.")
            except Exception as e:
                print(f"  Error converting to HF format: {e}")

        if os.path.isdir(hf_model_dir):
            quantized = load_compressed_model(hf_model_dir, prune_amount=0.0)
            results['quantized_int8'] = evaluate_compressed_model(quantized, val_loader, 'Quantized INT8', output_dir=output_dir)
            del quantized
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print("  Skipping quantization (model directory missing).")
            
    # 4. Combined Model (Pruned + INT8)
    if technique in ['combined', 'all']:
        if os.path.isdir(hf_model_dir):
            combined = load_compressed_model(hf_model_dir, prune_amount=0.3)
            results['combined'] = evaluate_compressed_model(combined, val_loader, 'Pruned + INT8', output_dir=output_dir)
            del combined
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print("  Skipping combined compression (model directory missing).")
    
    gc.collect()
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  COMPRESSION COMPARISON ({technique})")
    print(f"{'='*70}\n")
    
    comparison_df = pd.DataFrame({
        'Model': [r['name'] for r in results.values()],
        'Accuracy': [f"{r['accuracy']:.4f}" for r in results.values()],
        'Macro F1': [f"{r['macro_f1']:.4f}" for r in results.values()],
        'Size (MB)': [f"{r['size_mb']:.2f}" for r in results.values()],
        'Inference (ms)': [f"{r['inference_time_ms']:.3f}" for r in results.values()],
    })
    print(comparison_df.to_string(index=False))
    
    # Detailed Imbalance Analysis
    print(f"\n  Granular Analysis (Imbalance Handling):")
    for key, res in results.items():
        print(f"\n  [{res['name']}]")
        cr = res['per_class_report']
        # Check "Surprise" (Label 5) specifically
        surprise = cr.get('5', {})
        print(f"   Surprise (Class 5): Recall={surprise.get('recall', 0):.4f}, Precision={surprise.get('precision', 0):.4f}, F1={surprise.get('f1-score', 0):.4f}")
        print(f"   Macro F1: {res['macro_f1']:.4f}")
    
    # Calculate improvements
    orig = results['original']
    print(f"\n  Compression Analysis:")
    for key, res in results.items():
        if key == 'original': continue
        
        reduction = (res['size_mb'] / orig['size_mb'] - 1) * 100
        speedup = orig['inference_time_ms'] / res['inference_time_ms'] if res['inference_time_ms'] > 0 else 0
        f1_diff = res['macro_f1'] - orig['macro_f1']
        
        degraded = " DEGRADED" if f1_diff < -0.05 else ""
        print(f"   {res['name']:20s}: {reduction:+.1f}% size, {speedup:.2f}x speed, {f1_diff:+.4f} Macro F1{degraded}")
    
    # Save comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    names = [r['name'] for r in results.values()]
    accuracies = [r['accuracy'] for r in results.values()]
    macro_f1s = [r['macro_f1'] for r in results.values()]
    sizes = [r['size_mb'] for r in results.values()]
    times = [r['inference_time_ms'] for r in results.values()]
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    
    # Row 1: Accuracy and Macro F1
    axes[0, 0].bar(names, accuracies, color=colors[:len(names)])
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylim([min(accuracies) - 0.05, 1.0])
    axes[0, 0].tick_params(axis='x', rotation=15)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    axes[0, 1].bar(names, macro_f1s, color=colors[:len(names)])
    axes[0, 1].set_ylabel('Macro F1')
    axes[0, 1].set_title('Macro F1 Comparison')
    axes[0, 1].set_ylim([min(macro_f1s) - 0.05, 1.0])
    axes[0, 1].tick_params(axis='x', rotation=15)
    for i, v in enumerate(macro_f1s):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    # Row 2: Size and Speed
    axes[1, 0].bar(names, sizes, color=colors[:len(names)])
    axes[1, 0].set_ylabel('Size (MB)')
    axes[1, 0].set_title('Model Size Comparison')
    axes[1, 0].tick_params(axis='x', rotation=15)
    for i, v in enumerate(sizes):
        axes[1, 0].text(i, v + 5, f'{v:.1f}', ha='center', fontsize=9)
    
    axes[1, 1].bar(names, times, color=colors[:len(names)])
    axes[1, 1].set_ylabel('Inference Time (ms/sample)')
    axes[1, 1].set_title('Inference Speed Comparison')
    axes[1, 1].tick_params(axis='x', rotation=15)
    for i, v in enumerate(times):
        axes[1, 1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
    
    title_suffix = technique.capitalize() if technique != 'all' else "Pruning + INT8"
    plt.suptitle(f'Model Compression Results ({title_suffix})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    chart_name = f'{technique}_comparison.png' if technique != 'all' else 'compression_comparison.png'
    chart_path = os.path.join(vis_dir, chart_name)
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Chart saved: {chart_path}")
    
    # Save JSON report
    report = {
        'model_id': model_id,
        'weights_path': weights_path,
        'technique_filter': technique,
        'results': {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
    }
    
    report_name = f'{technique}_report.json' if technique != 'all' else 'compression_report.json'
    report_path = os.path.join(output_dir, report_name)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")
    
    return results


# ==================== MAIN TRAINING ====================
def main(train_path='train.csv', val_path='validation.csv', output_dir='./outputs', specific_model=None):
    """
    Main training pipeline.
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        output_dir: Directory for outputs (checkpoints, visualizations, reports)
        specific_model: Train only this model (None = train all)
    """
    models_to_run = MODELS
    if specific_model:
        if specific_model in MODELS:
            models_to_run = {specific_model: MODELS[specific_model]}
            print(f"Selected model: {specific_model}")
        else:
            raise ValueError(f"Model {specific_model} not found in {list(MODELS.keys())}")

    print(f"Models to train: {list(models_to_run.keys())}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    train_df, val_df = load_and_prepare_data(train_path, val_path)
    if train_df is None:
        raise Exception("Data not loaded")

    class_weights = get_class_weights(train_df['label'].values)
    print("\nClass distribution:\n", train_df['label'].value_counts().sort_index())

    models_results = {}
    
    # ==================== TRAINING LOOP ====================
    for model_name, model_id in models_to_run.items():
        print("\n" + "="*70)
        print(f"TRAINING: {model_name}")
        print("="*70)

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            clf = EmotionClassifier(model_name, model_id, DEVICE, class_weights=class_weights)

            train_ds = EmotionDataset(train_df['text'], train_df['label'], clf.tokenizer, MAX_LENGTH)
            val_ds = EmotionDataset(val_df['text'], val_df['label'], clf.tokenizer, MAX_LENGTH)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            print(f" Training: {len(train_ds)} | Validation: {len(val_ds)}")
            print(f" Batch size: {BATCH_SIZE} | Steps/epoch: {len(train_loader)}")

            clf.train(train_loader, val_loader, num_epochs=EPOCHS, output_dir=output_dir)

            preds, true_labels, logits, infer_time = clf.predict(val_loader)

            # Compute all metrics
            acc = accuracy_score(true_labels, preds)
            weighted_f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)
            macro_f1 = f1_score(true_labels, preds, average='macro', zero_division=0)
            macro_precision = precision_score(true_labels, preds, average='macro', zero_division=0)
            macro_recall = recall_score(true_labels, preds, average='macro', zero_division=0)
            cm = confusion_matrix(true_labels, preds)
            cr = classification_report(true_labels, preds, output_dict=True, zero_division=0)

            model_size = get_model_size_mb(clf.model)

            # Save confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=list(EMOTION_LABELS.values()),
                        yticklabels=list(EMOTION_LABELS.values()))
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'{model_name} - Confusion Matrix')
            cm_path = os.path.join(output_dir, 'visualizations', f'{model_name}_confusion_matrix.png')
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close()

            # Save training curves
            clf.save_training_curves(output_dir)

            models_results[model_name] = {
                'predictions': preds,
                'true_labels': true_labels,
                'accuracy': acc,
                'weighted_f1': weighted_f1,
                'macro_f1': macro_f1,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'confusion_matrix': cm.tolist(),
                'inference_time': infer_time,
                'model_size': model_size,
                'history': clf.history,
                'classification_report': cr
            }

            # Display results
            print(f"\n✅ {model_name} Results:")
            print(f" Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            print(f" Macro F1: {macro_f1:.4f}")
            print(f" Macro Precision: {macro_precision:.4f}")
            print(f" Macro Recall: {macro_recall:.4f}")
            print(f" Weighted F1: {weighted_f1:.4f}")
            print(f" Inference: {infer_time*1000:.2f} ms/sample")
            print(f" Size: {model_size:.2f} MB")

            # Per-class breakdown
            print("\n Per-class metrics:")
            for label_id, label_name in EMOTION_LABELS.items():
                cls = cr.get(str(label_id), {})
                print(f"   {label_name:10s}: P={cls.get('precision',0):.3f} "
                      f"R={cls.get('recall',0):.3f} F1={cls.get('f1-score',0):.3f}")

            del clf.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print("❌ Error training", model_name, ":", e)
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)

    # ==================== RESULTS SUMMARY ====================
    if models_results:
        print("\n📊 FINAL RESULTS\n")
        df = pd.DataFrame({
            'Model': list(models_results.keys()),
            'Accuracy': [models_results[m]['accuracy'] for m in models_results],
            'Macro F1': [models_results[m]['macro_f1'] for m in models_results],
            'Weighted F1': [models_results[m]['weighted_f1'] for m in models_results],
            'Inference (ms)': [models_results[m]['inference_time']*1000 for m in models_results],
            'Size (MB)': [models_results[m]['model_size'] for m in models_results],
        })
        print(df.to_string(index=False))

        # Best model by Macro F1 (appropriate for imbalanced data)
        best = max(models_results.keys(), key=lambda x: models_results[x]['macro_f1'])
        print(f"\n🏆 Best model (by Macro F1): {best}")
        print(f" Macro F1: {models_results[best]['macro_f1']:.4f}")
        print(f" Accuracy: {models_results[best]['accuracy']:.4f}")

        # Save comprehensive report
        report = {
            'best_model': best,
            'selection_metric': 'macro_f1',
            'models_comparison': {
                m: {
                    'accuracy': r['accuracy'],
                    'macro_f1': r['macro_f1'],
                    'macro_precision': r['macro_precision'],
                    'macro_recall': r['macro_recall'],
                    'weighted_f1': r['weighted_f1'],
                    'inference_time_ms': r['inference_time']*1000,
                    'model_size_mb': r['model_size'],
                    'confusion_matrix': r['confusion_matrix'],
                    'per_class_report': r['classification_report'],
                }
                for m, r in models_results.items()
            }
        }
        report_path = os.path.join(output_dir, 'emotion_classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved: {report_path}")
    else:
        print("No results - check errors above.")


# ==================== CLI INTERFACE ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Emotion Classification with BERT-based Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Training all models:
    python emotion_classifier.py --train train.csv --val validation.csv

  Training specific model on GPU 1:
    python emotion_classifier.py --model BERTweet --output ./outputs --gpu 1

  Inference on test set:
    python emotion_classifier.py --mode inference --weights best_ModernBERT.pt --test_csv test.csv

  Compression analysis:
    python emotion_classifier.py --mode compress --weights best_ModernBERT.pt --val validation.csv
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'inference', 'compress'],
                        help='Operation mode (default: train)')
    
    # Training arguments
    parser.add_argument('--train', type=str, default='train.csv',
                        help='Training CSV path')
    parser.add_argument('--val', type=str, default='validation.csv',
                        help='Validation CSV path')
    parser.add_argument('--output', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--model', type=str, default=None,
                        choices=list(MODELS.keys()),
                        help='Specific model to train')
    
    # Inference arguments
    parser.add_argument('--weights', type=str,
                        help='Path to .pt weights file')
    parser.add_argument('--test_csv', type=str,
                        help='Path to test CSV for inference')
    parser.add_argument('--model_id', type=str,
                        help='HuggingFace model ID (if not in checkpoint)')
    
    # Compression arguments
    parser.add_argument('--technique', type=str, default='all',
                        choices=['prune', 'quantize', 'combined', 'all'],
                        help='Compression technique to apply (prune, quantize, combined, or all)')
    
    # Device selection
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID (e.g., 0 or 1). Default: auto-detect')
    
    args = parser.parse_args()
    
    # Set device before anything else
    if args.gpu is not None:
        set_device(args.gpu)
    
    _print_config(args.gpu)
    
    if args.mode == 'train':
        print(f"\n📌 Mode: Training")
        print(f" Train: {args.train} | Val: {args.val} | Output: {args.output}")
        if args.model:
            print(f" Model: {args.model}")
        main(
            train_path=args.train,
            val_path=args.val,
            output_dir=args.output,
            specific_model=args.model
        )
    
    elif args.mode == 'inference':
        print(f"\n📌 Mode: Inference")
        if not args.weights or not args.test_csv:
            raise ValueError("--weights and --test_csv required for inference mode")
        predictions = run_inference(
            weights=args.weights,
            csv=args.test_csv,
            model_id=args.model_id
        )
        print(f"\n✅ Generated {len(predictions)} predictions")
    
    elif args.mode == 'compress':
        print(f"\n📌 Mode: Compression Analysis")
        if not args.weights:
            raise ValueError("--weights required for compress mode")
        compress_and_evaluate(
            weights_path=args.weights,
            val_csv=args.val,
            model_id=args.model_id,
            output_dir=args.output,
            technique=args.technique
        )
