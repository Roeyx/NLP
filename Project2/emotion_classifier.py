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
        python emotion_classifier.py --mode inference --weights outputs/best_CardiffRoBERTa.pt --test_csv test.csv

    Evaluation on labeled data (Accuracy/F1):
        python emotion_classifier.py --mode evaluate --weights outputs/best_CardiffRoBERTa.pt --val validation.csv

    Compression analysis:
        python emotion_classifier.py --mode compress --weights outputs/best_CardiffRoBERTa.pt --val validation.csv

Requirements:
    - Python >= 3.8
    - See requirements.txt for dependencies
    - GPU recommended (CUDA), CPU fallback supported
"""

import os
import gc
import re
import time
import json
import argparse
import warnings
import tempfile
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import emoji

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score, 
    precision_score, 
    recall_score,
    precision_recall_curve,
    average_precision_score
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Defaults
DEFAULT_TRAIN_PATH = 'train.csv'
DEFAULT_VAL_PATH = 'validation.csv'
DEFAULT_OUTPUT_DIR = './outputs'

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 5e-5
MAX_LENGTH = 128
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01


# ==================== UTILITIES ====================
def set_global_seed(seed=SEED):
    """Set random seeds for reproducibility across all libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_checkpoint_metadata(weights_path, device=None):
    """
    Robustly load checkpoint and extract metadata.
    Handles (metadata dict) formats.
    """
    device = device or 'cpu' # Safer to load metadata on CPU
    print(f"Loading checkpoint: {weights_path}")
    
    try:
        checkpoint = torch.load(weights_path, map_location=device)
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

    metadata = {
        'model_id': None,
        'config': {'max_length': MAX_LENGTH},
        'state_dict': None
    }

    # Checkpoint format assignment
    metadata['state_dict'] = checkpoint['model_state_dict']
    metadata['model_id'] = checkpoint.get('model_id')
    metadata['config'] = checkpoint.get('config', {})
    metadata['compression_type'] = checkpoint.get('compression_type') # Capture compression info
    print(f"  Format: Metadata-Rich (Model ID: {metadata['model_id']}, Compression: {metadata['compression_type']})")

    return metadata


def get_model_size_mb(model):
    """Calculate model size in MB by saving to a temp file."""
    with tempfile.NamedTemporaryFile() as tmp:
        torch.save(model.state_dict(), tmp.name)
        size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
    return size_mb


# ==================== VISUALIZATION UTILS ====================
class VisualizationUtils:
    """Centralized plotting utilities for creating figures."""
    
    @staticmethod
    def plot_class_distribution(labels, output_dir):
        """Plot bar chart of class distribution (Imbalance check)."""
        counts = Counter(labels)
        labels_ordered = [EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS.keys())]
        values_ordered = [counts[i] for i in sorted(EMOTION_LABELS.keys())]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels_ordered, values_ordered, color='#3498db')
        plt.title('Class Distribution (Training Data)', fontsize=14)
        plt.xlabel('Emotion')
        plt.ylabel('Number of Samples')
        plt.grid(axis='y', alpha=0.3)
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}', ha='center', va='bottom')
            
        out_path = os.path.join(output_dir, 'class_distribution.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Class distribution plot saved: {out_path}")

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(EMOTION_LABELS.values()),
                    yticklabels=list(EMOTION_LABELS.values()))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'{model_name} - Confusion Matrix')
        
        safe_name = re.sub(r'[^\w\-_]', '_', model_name)
        out_path = os.path.join(output_dir, f'{safe_name}_confusion_matrix.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📉 Confusion matrix saved: {out_path}")

    @staticmethod
    def plot_training_curves(history, model_name, output_dir):
        """Plot loss and accuracy curves."""
        epochs = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
        axes[0].plot(epochs, history['val_loss'], 'r-o', label='Val Loss')
        axes[0].set_title(f'{model_name} - Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, history['val_acc'], 'g-o', label='Val Acc')
        axes[1].set_title(f'{model_name} - Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        safe_name = re.sub(r'[^\w\-_]', '_', model_name)
        out_path = os.path.join(output_dir, f'{safe_name}_training_curves.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Training curves saved: {out_path}")

    @staticmethod
    def plot_pr_curves(y_true, y_logits, model_name, output_dir):
        """
        Plot Precision-Recall curves for all classes.
        Essential for imbalanced datasets.
        """
        # Convert true labels to one-hot
        n_classes = len(EMOTION_LABELS)
        y_true_onehot = np.eye(n_classes)[y_true]
        
        # Softmax logits to get probabilities
        y_probs = torch.softmax(torch.tensor(y_logits), dim=1).numpy()
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_probs[:, i])
            avg_precision = average_precision_score(y_true_onehot[:, i], y_probs[:, i])
            plt.plot(recall, precision, lw=2, 
                     label=f'{EMOTION_LABELS[i]} (AP={avg_precision:.2f})')
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} - Precision-Recall Curves')
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        safe_name = re.sub(r'[^\w\-_]', '_', model_name)
        out_path = os.path.join(output_dir, f'{safe_name}_pr_curves.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 PR Curves saved: {out_path}")


# ==================== DATA CLEANER ====================
class DataCleaner:
    """Utilities to clean and filter raw text data."""
    
    @staticmethod
    def clean_text(text):
        """Remove URLs, emails, mentions, hashtags, non-ascii chars."""
        if not isinstance(text, str):
            return ""
        
        # 1. Demojize: Convert emojis to text (e.g. 😂 -> " joy ")
        text = emoji.demojize(text, delimiters=(" ", " "))
        
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # 2. ASCII normalization (respects converted emojis)
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    @staticmethod
    def remove_duplicates(df):
        initial = len(df)
        df = df.drop_duplicates(subset=['text'], keep='first')
        print(f" Removed {initial - len(df)} duplicates")
        return df

    @staticmethod
    def remove_outliers(df, min_len=3, max_len=512):
        initial = len(df)
        df = df[(df['text'].str.len() >= min_len) & (df['text'].str.len() <= max_len)]
        print(f" Removed {initial - len(df)} outliers")
        return df

    @staticmethod
    def handle_missing_values(df):
        return df.dropna(subset=['text', 'label'])


# ==================== DATASETS ====================
class EmotionDataset(Dataset):
    """PyTorch Dataset for training/validation (with labels)."""
    
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
    """PyTorch Dataset for inference (test dataset)."""
    
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


# ==================== CLASSIFIER CLASS ====================
class EmotionClassifier:
    """Encapsulates model training and evaluation logic."""
    
    def __init__(self, model_name, model_id, device, class_weights=None):
        self.model_name = model_name
        self.model_id = model_id
        self.device = device
        self.class_weights = class_weights

        print(f"\nInitializing {model_name} ({model_id})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=6,
            ignore_mismatched_sizes=True
        )

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f" Parameters: {trainable:,} trainable")

        self.model.to(device)
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.best_val_loss = float('inf')

    def get_loss_fn(self):
        """Active Imbalance Handling: Use Weighted Cross Entropy."""
        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, dtype=torch.float).to(self.device)
            return nn.CrossEntropyLoss(weight=weights)
        return nn.CrossEntropyLoss()

    def train_epoch(self, train_loader, optimizer, scheduler, loss_fn):
        self.model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Train {self.model_name}", leave=False):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate(self, val_loader, loss_fn):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Eval {self.model_name}", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        return total_loss / len(val_loader), acc, all_preds, all_labels

    def train(self, train_loader, val_loader, num_epochs=5, output_dir='./outputs'):
        """Train loop with checkpointing and metadata saving."""
        loss_fn = self.get_loss_fn()
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=WARMUP_STEPS, 
            num_training_steps=len(train_loader) * num_epochs
        )

        patience = 3
        patience_counter = 0
        ckpt_path = os.path.join(output_dir, f'best_{self.model_name}.pt')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            train_loss = self.train_epoch(train_loader, optimizer, scheduler, loss_fn)
            val_loss, val_acc, _, _ = self.evaluate(val_loader, loss_fn)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f" Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                
                # Save robust checkpoint
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'model_id': self.model_id,
                    'model_name': self.model_name,
                    'config': {'max_length': MAX_LENGTH},
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, ckpt_path)
                
                # Save HuggingFace format for Quantization support
                hf_dir = ckpt_path.replace('.pt', '')
                self.model.save_pretrained(hf_dir)
                self.tokenizer.save_pretrained(hf_dir)
                print(f" Saved Best Model: {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(" Early stopping triggered.")
                    break

        # Reload best weights
        meta = load_checkpoint_metadata(ckpt_path, self.device)
        self.model.load_state_dict(meta['state_dict'])


# ==================== INFERENCE ====================
def run_inference(weights: str, csv: str, model_id: str = None, 
                  text_column: str = 'text', output_csv: str = None) -> list:
    """
    Inference Interface.
    """
    set_global_seed(SEED)
    print(f"\n{'='*40}\nRUNNING INFERENCE\n{'='*40}")
    
    # 1. Load Metadata & Model
    meta = load_checkpoint_metadata(weights, DEVICE)
    
    # Use stored model_id if available, otherwise argument
    resolved_model_id = meta['model_id'] or model_id
    if not resolved_model_id:
        raise ValueError("Model ID not found in checkpoint. Provide --model_id.")
    
    print(f" Model ID: {resolved_model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        resolved_model_id, num_labels=6, ignore_mismatched_sizes=True
    )
    model.load_state_dict(meta['state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # 2. Prepare Data
    try:
        df = pd.read_csv(csv)
    except FileNotFoundError:
        print(f"Error: File {csv} not found.")
        return []
        
    print(f" Data: {len(df)} rows from {csv}")
    
    # Clean and Load
    df[text_column] = df[text_column].apply(DataCleaner.clean_text)
    dataset = InferenceDataset(df[text_column], tokenizer, meta['config'].get('max_length', 128))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Predict
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds = torch.argmax(outputs.logits, dim=1)
            preds.extend(batch_preds.cpu().numpy().tolist())
            
    # 4. Save
    if output_csv is None:
        base = os.path.splitext(os.path.basename(csv))[0]
        output_csv = f"{base}_predictions.csv"
        
    result_df = df.copy()
    result_df['predicted_label'] = preds
    result_df['predicted_emotion'] = [EMOTION_LABELS[p] for p in preds]
    result_df.to_csv(output_csv, index=False)
    print(f" Predictions saved to: {output_csv}")
    
    return preds


# ==================== EVALUATION ====================
def run_evaluation(weights_path: str, val_csv: str, model_id: str = None, output_dir: str = './outputs'):
    """
    Evaluate a trained model on a labeled dataset.
    Calculates Accuracy, F1, and generates Confusion Matrix.
    """
    set_global_seed(SEED)
    print(f"\n{'='*40}\nRUNNING EVALUATION\n{'='*40}")
    
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # 1. Load Metadata
    meta = load_checkpoint_metadata(weights_path, 'cpu')
    resolved_model_id = meta['model_id'] or model_id
    if not resolved_model_id:
        raise ValueError("Model ID not found in checkpoint. Provide --model_id.")
    
    print(f" Model ID: {resolved_model_id}")
    
    # 2. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_id)
    
    # Check for compression metadata
    compression = meta.get('compression_type', 'none')
    print(f" Detected Compression: {compression}")

    if compression in ['quantized_int8', 'quantized_nf4', 'combined']:
        # Map compression string to parameters
        quant_type = 'int8' if 'int8' in compression else 'nf4'
        prune_amt = 0.3 if 'combined' in compression else 0.0
        
        # Load architecture with quantization config
        # We assume the base model ID is sufficient to reconstruct the architecture
        model = load_compressed_model(resolved_model_id, prune_amt, quant_type, DEVICE)
        
        # Determine strictness: Quantized state dicts often have missing keys 
        # (e.g. .absmax, .quant_map) that are buffers, so strict=False might be needed.
        # However, for this project, we want to ensure we are evaluating the *saved* weights.
        try:
            model.load_state_dict(meta['state_dict'], strict=False)
            print(" Loaded quantized state_dict (strict=False)")
        except Exception as e:
            print(f" Warning: Could not load state_dict directly: {e}")
            print(" Using re-quantized base model (deterministic fallback).")
            
    elif compression == 'pruned_30':
        model = AutoModelForSequenceClassification.from_pretrained(resolved_model_id, num_labels=6)
        model = apply_pruning(model, 0.3)
        model.load_state_dict(meta['state_dict'])
        model.to(DEVICE)
        
    else:
        # Standard FP32 loading
        model = AutoModelForSequenceClassification.from_pretrained(
            resolved_model_id, num_labels=6, ignore_mismatched_sizes=True
        )
        model.load_state_dict(meta['state_dict'])
        model.to(DEVICE)

    model.eval()

    # 3. Load Data
    print(f" Loading data from {val_csv}...")
    try:
        df = pd.read_csv(val_csv)
    except FileNotFoundError:
        print(f"Error: File {val_csv} not found.")
        return

    # Clean
    df['text'] = df['text'].apply(DataCleaner.clean_text)
    
    # Dataset (Expects 'label' column)
    if 'label' not in df.columns:
        raise ValueError(f"CSV {val_csv} must contain a 'label' column for evaluation.")

    dataset = EmotionDataset(df['text'], df['label'], tokenizer, meta['config'].get('max_length', 128))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Evaluate
    results = evaluate_model_performance(model, loader, "Eval_Model", device=DEVICE, output_dir=vis_dir)

    # 5. Print Report
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f" Accuracy:  {results['accuracy']:.4f}")
    print(f" Macro F1:  {results['macro_f1']:.4f}")
    print(f" Latency:   {results['inference_time_ms']:.2f} ms/sample")
    print("-" * 20)
    print("Per-Class Report:")
    report_df = pd.DataFrame(results['per_class_report']).transpose()
    print(report_df.to_string())
    print("-" * 20)
    print(f"Plots saved to: {vis_dir}")

    return results


# ==================== COMPRESSION ====================
def apply_pruning(model, amount=0.3):
    """Apply unstructured L1 pruning to Linear layers."""
    pruned_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
            pruned_count += 1
    print(f"  Pruned {pruned_count} Linear layers at {amount*100:.0f}% sparsity")
    return model


def evaluate_model_performance(model, val_loader, name, device=None, output_dir=None):
    """
    Unified evaluation function for Training, Compression, and Evaluation modes.
    Handles device placement (standard/quantized), metrics, and visualizations.
    """
    device = device or DEVICE
    
    # 1. Smart Device Placement
    # If model is 8-bit/4-bit, it is already mapped. Do not move.
    is_quantized = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    if not is_quantized:
        model.to(device)
        target_device = device
    else:
        # For quantized models, inputs must be on the same device as the model's execution
        target_device = model.device

    model.eval()
    
    all_preds, all_labels, all_logits = [], [], []
    total_time = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Eval {name}", leave=False):
            # Move inputs to the correct device
            input_ids = batch['input_ids'].to(target_device)
            attention_mask = batch['attention_mask'].to(target_device)
            labels = batch['labels']
            
            start = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            total_time += time.time() - start
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_logits.extend(logits.cpu().numpy())
    
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    size_mb = get_model_size_mb(model)
    avg_time_ms = (total_time / len(all_preds)) * 1000
    
    # Visualizations
    if output_dir:
        VisualizationUtils.plot_confusion_matrix(all_labels, all_preds, name, output_dir)
        VisualizationUtils.plot_pr_curves(all_labels, all_logits, name, output_dir)
    
    return {
        'name': name,
        'accuracy': acc,
        'macro_f1': macro_f1,
        'size_mb': size_mb,
        'inference_time_ms': avg_time_ms,
        'per_class_report': classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    }


def load_compressed_model(model_dir: str, prune_amount: float = 0.0, quant_type: str = 'int8', device=None):
    """Load model with Quantization (INT8 or NF4), optionally with Pruning."""
    label = f"{'Pruned ' if prune_amount > 0 else ''}{quant_type.upper()}"
    print(f"\n  Creating Compressed Model ({label})...")
        
    from transformers import BitsAndBytesConfig
    
    # 1. Load fine-tuned weights (FP32) 
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=6, ignore_mismatched_sizes=True
    )
    
    # 2. Pruning
    if prune_amount > 0:
        model = apply_pruning(model, prune_amount)
    
    # Save temp model for reloading, necessary for BitsAndBytesConfig since it required loading from a directory
    temp_dir = "./temp_compressed_model"
    model.save_pretrained(temp_dir)
    del model
    gc.collect()
    
    # 3. Quantization Config
    if quant_type == 'nf4':
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    else:  # int8 skipping classification head to avoid collapse of the softmax layer
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=["classifier"]
        )
    
    # Determine device map to force single-GPU execution
    if device and device.type == 'cuda':
        d_map = {"": device.index if device.index is not None else 0}
    else:
        d_map = "auto"

    # Reload
    compressed_model = AutoModelForSequenceClassification.from_pretrained(
        temp_dir,
        quantization_config=quant_config,
        device_map=d_map
    )
    
    # Cleanup temporary files 
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        
    return compressed_model


def compress_and_evaluate(weights_path: str, val_csv: str, model_id: str = None,
                          output_dir: str = './compression_results',
                          technique: str = 'all') -> dict:
    """Run compression benchmarks."""
    set_global_seed(SEED)
    print(f"\n{'='*40}\nCOMPRESSION ANALYSIS ({technique})\n{'='*40}")
    
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # 1. Load Metadata
    meta = load_checkpoint_metadata(weights_path, 'cpu')
    resolved_model_id = meta['model_id'] or model_id
    if not resolved_model_id:
        raise ValueError("Model ID required.")
    
    # Ensure HF format exists
    hf_dir = weights_path.replace('.pt', '')
    if not os.path.isdir(hf_dir):
        print(f" Converting to HF format at {hf_dir}...")
        temp = AutoModelForSequenceClassification.from_pretrained(resolved_model_id, num_labels=6, ignore_mismatched_sizes=True)
        temp.load_state_dict(meta['state_dict'])
        temp.save_pretrained(hf_dir)
        AutoTokenizer.from_pretrained(resolved_model_id).save_pretrained(hf_dir)
        del temp

    # 2. Data
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_id)
    val_df = pd.read_csv(val_csv)
    val_df['text'] = val_df['text'].apply(DataCleaner.clean_text)
    val_ds = EmotionDataset(val_df['text'], val_df['label'], tokenizer, MAX_LENGTH)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    results = {}

    def save_variant(model, suffix):
        """Helper to save compressed model variants."""
        base_name = os.path.splitext(os.path.basename(weights_path))[0]
        fname = f"{base_name}_{suffix}.pt"
        save_path = os.path.join(output_dir, fname)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_id': resolved_model_id,
            'config': meta.get('config', {}),
            'compression_type': suffix
        }, save_path)
        print(f"  💾 Saved {suffix} model to: {save_path}")
        return save_path

    # --- Benchmark 1: Original ---
    print("\n1. Evaluating Original...")
    orig = AutoModelForSequenceClassification.from_pretrained(resolved_model_id, num_labels=6, ignore_mismatched_sizes=True)
    orig.load_state_dict(meta['state_dict'])
    results['original'] = evaluate_model_performance(orig, val_loader, "Original", device=DEVICE, output_dir=vis_dir)
    del orig
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Benchmark 2: Pruned ---
    if technique in ['prune', 'all', 'combined']:
        print("\n2. Evaluating Pruned (30%)...")
        pruned = AutoModelForSequenceClassification.from_pretrained(resolved_model_id, num_labels=6, ignore_mismatched_sizes=True)
        pruned.load_state_dict(meta['state_dict'])
        pruned = apply_pruning(pruned, 0.3)
        # Save first
        save_path = save_variant(pruned, "pruned_30")
        # Reload to ensure consistency
        pruned.load_state_dict(torch.load(save_path)['model_state_dict'])
        results['pruned'] = evaluate_model_performance(pruned, val_loader, "Pruned 30%", device=DEVICE, output_dir=vis_dir)
        del pruned
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Benchmark 3: Quantized ---
    if technique in ['quantize', 'all', 'combined']:
        if os.path.isdir(hf_dir):
            # INT8
            print("\n3a. Evaluating Quantized (INT8)...")
            q8 = load_compressed_model(hf_dir, 0.0, 'int8', device=DEVICE)
            # Save first
            save_path = save_variant(q8, "quantized_int8")
            # Reload to ensure consistency (simulating user evaluation)
            q8.load_state_dict(torch.load(save_path)['model_state_dict'], strict=False)
            results['quantized_int8'] = evaluate_model_performance(q8, val_loader, "Quantized INT8", output_dir=vis_dir)
            del q8
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            # NF4
            print("\n3b. Evaluating Quantized (NF4)...")
            q4 = load_compressed_model(hf_dir, 0.0, 'nf4', device=DEVICE)
            # Save first
            save_path = save_variant(q4, "quantized_nf4")
            # Reload to ensure consistency (simulating user evaluation)
            q4.load_state_dict(torch.load(save_path)['model_state_dict'], strict=False)
            results['quantized_nf4'] = evaluate_model_performance(q4, val_loader, "Quantized NF4", output_dir=vis_dir)
            del q4
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Benchmark 4: Combined pruning and nf4 ---
    if technique in ['combined', 'all']:
        if os.path.isdir(hf_dir):
            print("\n4. Evaluating Combined (Pruned + NF4)...")
            comb = load_compressed_model(hf_dir, 0.3, 'nf4', device=DEVICE)
            # Save first
            save_path = save_variant(comb, "combined")
            # Reload to ensure consistency (simulating user evaluation)
            comb.load_state_dict(torch.load(save_path)['model_state_dict'], strict=False)
            results['combined'] = evaluate_model_performance(comb, val_loader, "Pruned + NF4", output_dir=vis_dir)
            del comb
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Report
    print("\n--- Compression Results ---")
    rows = []
    for k, v in results.items():
        rows.append({
            'Model': v['name'],
            'Acc': v['accuracy'],
            'MacroF1': v['macro_f1'],
            'Size(MB)': v['size_mb'],
            'Lat(ms)': v['inference_time_ms']
        })
    print(pd.DataFrame(rows).to_string(index=False))
    
    with open(os.path.join(output_dir, 'compression_report.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    return results


# ==================== MAIN TRAINING LOOP ====================
def main(train_path=DEFAULT_TRAIN_PATH, val_path=DEFAULT_VAL_PATH, output_dir=DEFAULT_OUTPUT_DIR, specific_model=None):
    """
    Main training pipeline.
    
    Args:
        train_path (str): Path to the training dataset CSV file.
        val_path (str): Path to the validation dataset CSV file.
        output_dir (str): Directory to save model checkpoints and visualizations.
        specific_model (str, optional): Name of a specific model to train (e.g., 'BERTweet').
                                        If None, trains all defined models.
    """
    set_global_seed(SEED)
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading data from {train_path}...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Visualize Imbalance 
    VisualizationUtils.plot_class_distribution(train_df['label'], vis_dir)
    
    cleaner = DataCleaner()
    for df in [train_df, val_df]:
        df = cleaner.handle_missing_values(df)
        df['text'] = df['text'].apply(cleaner.clean_text)
        df = cleaner.remove_duplicates(df)
        df = cleaner.remove_outliers(df)
        
    # Class Weights (Active Strategy)
    counts = Counter(train_df['label'])
    total = len(train_df)
    weights = [total / counts[i] for i in range(6)]
    class_weights = np.array(weights) / sum(weights) * 6
    print(f"Class Weights: {np.round(class_weights, 3)}")

    # 2. Train Models
    models_to_run = {specific_model: MODELS[specific_model]} if specific_model else MODELS
    results = {}

    for name, mid in models_to_run.items():
        print(f"\n--- Training {name} ---")
        
        clf = EmotionClassifier(name, mid, DEVICE, class_weights)
        train_ds = EmotionDataset(train_df['text'], train_df['label'], clf.tokenizer, MAX_LENGTH)
        val_ds = EmotionDataset(val_df['text'], val_df['label'], clf.tokenizer, MAX_LENGTH)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        clf.train(train_loader, val_loader, EPOCHS, output_dir)
        VisualizationUtils.plot_training_curves(clf.history, name, vis_dir)
        
        # Final Evaluation
        eval_results = evaluate_model_performance(clf.model, val_loader, name, device=DEVICE, output_dir=vis_dir)
        
        results[name] = {
            'accuracy': eval_results['accuracy'],
            'macro_f1': eval_results['macro_f1'],
            'inference_time': eval_results['inference_time_ms'] / 1000.0, # convert ms back to s
            'model_size': eval_results['size_mb']
        }
        
        print(f"Result {name}: Acc={eval_results['accuracy']:.4f}, MacroF1={eval_results['macro_f1']:.4f}")
        
        del clf
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Save Final Report
    best_model = max(results, key=lambda x: results[x]['macro_f1'])
    print(f"\n🏆 Best Model (Macro F1): {best_model}")
    
    with open(os.path.join(output_dir, 'emotion_classification_report.json'), 'w') as f:
        json.dump({'results': results, 'best_model': best_model}, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Emotion Classifier CLI")
    parser.add_argument('--mode', choices=['train', 'inference', 'compress', 'evaluate'], default='train', 
                        help="Operation mode: 'train', 'inference', 'compress', or 'evaluate'.")
    parser.add_argument('--train', default=DEFAULT_TRAIN_PATH, 
                        help=f"Path to training CSV file (default: {DEFAULT_TRAIN_PATH}).")
    parser.add_argument('--val', default=DEFAULT_VAL_PATH, 
                        help=f"Path to validation CSV file (default: {DEFAULT_VAL_PATH}).")
    parser.add_argument('--test_csv', help="Path to test CSV file. Required for inference mode.")
    parser.add_argument('--weights', help="Path to trained model checkpoint (.pt). Required for inference/compress/evaluate.")
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR, 
                        help=f"Directory to save outputs (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument('--model', choices=MODELS.keys(), 
                        help="Specific model architecture to use (optional). If omitted, trains all models.")
    parser.add_argument('--technique', default='all', choices=['prune', 'quantize', 'combined', 'all'],
                        help="Compression technique to benchmark: 'prune', 'quantize', 'combined', or 'all'.")
    parser.add_argument('--model_id', help="Hugging Face Model ID (e.g. 'vinai/bertweet-base'). Overrides metadata in checkpoint.")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        main(args.train, args.val, args.output, args.model)
    elif args.mode == 'inference':
        if not args.weights or not args.test_csv:
            raise ValueError("Inference requires --weights and --test_csv")
        run_inference(args.weights, args.test_csv, args.model_id)
    elif args.mode == 'evaluate':
        if not args.weights or not args.val:
            raise ValueError("Evaluation requires --weights and --val")
        run_evaluation(args.weights, args.val, args.model_id, args.output)
    elif args.mode == 'compress':
        if not args.weights or not args.val:
            raise ValueError("Compression requires --weights and --val")
        compress_and_evaluate(args.weights, args.val, args.model_id, args.output, args.technique)
