#!/usr/bin/env python3
"""
Emotion Classification with BERT-based Models
Models: BERTweet, CardiffRoBERTa, ModernBERT
Task: 6-class emotion classification
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==================== CONFIGURATION ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

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

print("\n⚙️ Configuration:")
print(f" Batch Size: {BATCH_SIZE}")
print(f" Max Length: {MAX_LENGTH}")
print(f" Epochs: {EPOCHS}")
print(f" Learning Rate: {LEARNING_RATE}")
print(f" Warmup Steps: {WARMUP_STEPS}")
print(f" Weight Decay: {WEIGHT_DECAY}")


# ==================== DATA CLEANER ====================
class DataCleaner:
    @staticmethod
    def clean_text(text):
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


# ==================== EMOTION DATASET ====================
class EmotionDataset(Dataset):
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


# ==================== EMOTION CLASSIFIER ====================
class EmotionClassifier:
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
        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, dtype=torch.float).to(self.device)
            return nn.CrossEntropyLoss(weight=weights)
        return nn.CrossEntropyLoss()

    def train_epoch(self, train_loader, optimizer, scheduler, loss_fn):
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

    def train(self, train_loader, val_loader, num_epochs=5):
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

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs} - {self.model_name}")
            train_loss = self.train_epoch(train_loader, optimizer, scheduler, loss_fn)
            val_loss, val_acc, _, _ = self.evaluate(val_loader, loss_fn)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), f'best_{self.model_name}.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping (patience={patience})")
                    break

        self.train_time = time.time() - start_time
        self.model.load_state_dict(torch.load(f'best_{self.model_name}.pt'))

    def predict(self, test_loader):
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
        return all_preds, all_labels, all_logits, avg_time


# ==================== DATA LOADING HELPERS ====================
def load_and_prepare_data(train_path='train.csv', val_path='validation.csv'):
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
    except Exception as e:
        print("Error: CSV files not found!", e)
        return None, None

    print(f"Original: {len(train_df)} train, {len(val_df)} val")
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
    counts = Counter(labels)
    total = len(labels)
    weights = [total / counts[i] for i in range(6)]
    weights = np.array(weights)
    weights = weights / weights.sum() * 6
    return weights


# ==================== MAIN TRAINING ====================
def main(train_path='train.csv', val_path='validation.csv', output_dir='./outputs'):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

    train_df, val_df = load_and_prepare_data(train_path, val_path)
    if train_df is None:
        raise Exception("Data not loaded")

    class_weights = get_class_weights(train_df['label'].values)
    print("\nClass weights:", class_weights)
    print("\nClass distribution:\n", train_df['label'].value_counts().sort_index())

    models_results = {}

    for model_name, model_id in MODELS.items():
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

            clf.train(train_loader, val_loader, num_epochs=EPOCHS)

            preds, true_labels, logits, infer_time = clf.predict(val_loader)

            acc = accuracy_score(true_labels, preds)
            f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)
            cr = classification_report(true_labels, preds, output_dict=True, zero_division=0)

            param_size = sum(p.numel() * p.element_size() for p in clf.model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in clf.model.buffers())
            model_size = (param_size + buffer_size) / 1024 / 1024

            models_results[model_name] = {
                'predictions': preds,
                'true_labels': true_labels,
                'logits': logits,
                'accuracy': acc,
                'f1_score': f1,
                'inference_time': infer_time,
                'model_size': model_size,
                'history': clf.history,
                'classification_report': cr
            }

            print(f"\n✅ {model_name} Results:")
            print(f" Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            print(f" F1-Score: {f1:.4f}")
            print(f" Inference: {infer_time*1000:.2f} ms")
            print(f" Size: {model_size:.2f} MB")

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
            'F1-Score': [models_results[m]['f1_score'] for m in models_results],
            'Inference (ms)': [models_results[m]['inference_time']*1000 for m in models_results],
            'Size (MB)': [models_results[m]['model_size'] for m in models_results],
        })
        print(df.to_string(index=False))

        best = max(models_results.keys(), key=lambda x: models_results[x]['accuracy'])
        print(f"\n🏆 Best model: {best}")
        print(f" Accuracy: {models_results[best]['accuracy']:.4f} ({models_results[best]['accuracy']*100:.2f}%)")

        report = {
            'best_model': best,
            'models_comparison': {
                m: {
                    'accuracy': r['accuracy'],
                    'f1_score': r['f1_score'],
                    'inference_time_ms': r['inference_time']*1000,
                    'model_size_mb': r['model_size'],
                }
                for m, r in models_results.items()
            }
        }
        report_path = os.path.join(output_dir, 'emotion_classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Report saved: {report_path}")
    else:
        print("No results to display – check for errors above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Emotion Classification Training')
    parser.add_argument('--train', type=str, default='train.csv', help='Path to training CSV')
    parser.add_argument('--val', type=str, default='validation.csv', help='Path to validation CSV')
    parser.add_argument('--output', type=str, default='./outputs', help='Output directory')

    args = parser.parse_args()

    main(train_path=args.train, val_path=args.val, output_dir=args.output)
