"""
Bidirectional GRU Model for Emotion Classification

"""

import os
import time

# CRITICAL: Set LD_LIBRARY_PATH before importing TensorFlow
os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('CONDA_PREFIX', '')}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import tensorflow as tf
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
import kagglehub

# Configuration Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
VAL_PATH = os.path.join(BASE_DIR, "validation.csv")

# Download latest version from Kaggle
path = kagglehub.dataset_download("bertcarremans/glovetwitter27b100dtxt")
print("Path to dataset files:", path)

# The dataset folder contains 'glove.twitter.27B.100d.txt'
GLOVE_FILE = os.path.join(path, "glove.twitter.27B.100d.txt")
print("Using GloVe file:", GLOVE_FILE)

MAX_WORDS = 30000
MAX_LEN = 100 
EMBED_DIM = 100  

GRID_WARMUP_EPOCHS = 2
GRID_FINETUNE_EPOCHS = 3
FINAL_WARMUP_EPOCHS = 5
FINAL_FINETUNE_EPOCHS = 15

LABEL_MAP = {
    0: 'sadness', 1: 'joy', 2: 'love',
    3: 'anger', 4: 'fear', 5: 'surprise'
}

# Grid: 2 units × 2 dropout × 4 LRs = 16 combos
PARAM_GRID = {
    'gru_units': [128, 256],
    'dropout': [0.2, 0.3],
    'fine_tune_lr': [1e-3, 5e-4, 1e-4, 5e-5],
    'warmup_lr': [1e-3], 
    'spatial_dropout': [0.2],
    'dense_units': [64],
}


def clean_text(text):
    """Clean and normalize text (aligned with LSTM.py)."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_embedding_matrix(tokenizer_word_index, glove_path=GLOVE_FILE, embed_dim=EMBED_DIM, max_words=MAX_WORDS):
    """Build embedding matrix from GloVe file (no caching, inline builder)."""
    print(f"Loading GloVe vectors from {glove_path}...")
    embeddings_index = {}
    
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading GloVe"):
            parts = line.rstrip().split()
            if len(parts) <= 2:
                continue
            word = parts[0]
            try:
                coefs = np.asarray(parts[1:], dtype='float32')
                if coefs.shape[0] == embed_dim:
                    embeddings_index[word] = coefs
            except ValueError:
                continue
    
    print(f"Loaded {len(embeddings_index)} word vectors")
    
    vocab_size = min(max_words, len(tokenizer_word_index) + 1)
    embedding_matrix = np.zeros((vocab_size, embed_dim), dtype='float32')
    
    hits = 0
    misses = 0
    for word, idx in tokenizer_word_index.items():
        if idx >= vocab_size:
            continue
        vec = embeddings_index.get(word)
        if vec is not None:
            #If not found, that row stays all zeros (unknown in GloVe).
            embedding_matrix[idx] = vec
            hits += 1
        else:
            misses += 1
    
    print(f"Embedding matrix built: {hits} hits, {misses} misses (coverage: {hits/max(vocab_size,1):.2%})")
    return embedding_matrix


class GRUModel:
    """Bidirectional GRU model for text classification."""
    
    def __init__(self, vocab_size, embedding_matrix, gru_units=64, dropout=0.5,
                 spatial_dropout=0.2, dense_units=64):
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        self.gru_units = gru_units
        self.dropout = dropout
        self.spatial_dropout = spatial_dropout
        self.dense_units = dense_units
        self.model = None
    
    def build_model(self):
        """Build the bidirectional GRU model architecture."""
        model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_matrix.shape[1],
                weights=[self.embedding_matrix],
                input_length=MAX_LEN,
                trainable=True,  # Enable fine-tuning
                mask_zero=True
            ),
            SpatialDropout1D(self.spatial_dropout),
            Bidirectional(GRU(self.gru_units, return_sequences=False)),
            Dense(self.dense_units, activation='relu'),
            Dropout(self.dropout),
            Dense(len(LABEL_MAP), activation='softmax')
        ])
        
        self.model = model
        return model
    
    def train_freeze_thaw(self, X_train, y_train, X_val, y_val, 
                         warmup_epochs, finetune_epochs, warmup_lr, fine_tune_lr, batch_size):
         """Train model using freeze-thaw strategy."""
         if self.model is None:
             self.build_model()
         
         # Phase 1: Warmup with frozen embeddings
         self.model.layers[0].trainable = False
         self.model.compile(
             optimizer=Adam(learning_rate=warmup_lr),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy']
         )
         
         history1 = self.model.fit(
             X_train, y_train,
             validation_data=(X_val, y_val),
             epochs=warmup_epochs,
             batch_size=batch_size,
             verbose=1
         )
         
         # Phase 2: Fine-tune with unfrozen embeddings
         self.model.layers[0].trainable = True
         self.model.compile(
            optimizer=Adam(learning_rate=fine_tune_lr),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy']
         )
         
         callbacks = [
             EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True),
             ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=1e-6)
         ]
         
         history2 = self.model.fit(
             X_train, y_train,
             validation_data=(X_val, y_val),
             epochs=finetune_epochs,
             batch_size=batch_size,
             callbacks=callbacks,
             verbose=1
         )
         
         return history1, history2
    
    def evaluate(self, X_val, y_val):
        """Evaluate model on validation data."""
        loss, acc = self.model.evaluate(X_val, y_val, verbose=0)
        return loss, acc
    
    def predict(self, X, **kwargs):
        """Make predictions."""
        return self.model.predict(X, **kwargs)


class GRUTrainer:
    """Manages training pipeline including grid search and final training."""
    
    def __init__(self, param_grid=PARAM_GRID):
        self.param_grid = param_grid
        self.best_params = None
        self.best_model = None
        self.grid_results = []
        
        # GPU setup: batch_size 256 if GPU else 32
        gpus = tf.config.list_physical_devices('GPU')
        self.has_gpu = len(gpus) > 0
        
        if self.has_gpu:
            print(f"✅ GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            self.batch_size = 256
            print(f"✓ Batch size set to {self.batch_size} (GPU-optimized)")
        else:
            print(f"⚠️ No GPU detected. Running on CPU.")
            self.batch_size = 32
            print(f"✓ Batch size set to {self.batch_size} (CPU-optimized)")
    
    def load_data(self, train_path=TRAIN_PATH, val_path=VAL_PATH):
        """Load and preprocess data (Keras Tokenizer like LSTM.py)."""
        print("\n=== Loading Data ===")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        train_df['clean_text'] = train_df['text'].apply(clean_text)
        val_df['clean_text'] = val_df['text'].apply(clean_text)
        
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        tokenizer.fit_on_texts(train_df['clean_text'])
        
        X_train_seq = tokenizer.texts_to_sequences(train_df['clean_text'])
        X_val_seq = tokenizer.texts_to_sequences(val_df['clean_text'])
        
        X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
        X_val = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # Encode labels with LabelEncoder (match LSTM.py)
        le = LabelEncoder()
        y_train = le.fit_transform(train_df['label'])
        y_val = le.transform(val_df['label'])
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Tokenizer vocab size: {min(MAX_WORDS, len(tokenizer.word_index) + 1)}")
        
        return X_train, y_train, X_val, y_val, tokenizer
    
    def run_grid_search(self, X_train, y_train, X_val, y_val, embedding_matrix, vocab_size):
        """Run grid search and store results in-memory."""
        grid = list(ParameterGrid(self.param_grid))
        total = len(grid)
        
        print(f"\n=== Grid Search: {total} combinations ===")
        print("⏳ First iteration may be slow (GPU warmup)...")
        
        best_acc = 0.0
        best_params = None
        self.grid_results = []
        
        for i, params in enumerate(grid, 1):
            print(f"\n[{i}/{total}] {params}")
            start = time.time()
            
            model = GRUModel(
                vocab_size, embedding_matrix,
                gru_units=params['gru_units'],
                dropout=params['dropout'],
                spatial_dropout=params['spatial_dropout'],
                dense_units=params['dense_units']
            )
            model.build_model()
            
            model.train_freeze_thaw(
                X_train, y_train, X_val, y_val,
                GRID_WARMUP_EPOCHS, GRID_FINETUNE_EPOCHS,
                params.get('warmup_lr', 1e-3), params['fine_tune_lr'], self.batch_size
            )
            
            _, val_acc = model.evaluate(X_val, y_val)
            elapsed = time.time() - start
            
            result = {
                'params': params,
                'val_acc': float(val_acc),
                'time_s': float(elapsed)
            }
            self.grid_results.append(result)
            
            print(f"  → Val Acc: {val_acc:.4f} (took {elapsed:.1f}s)")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = params
        
        self.best_params = best_params
        self._print_grid_summary()
        
        return best_params
    
    def _print_grid_summary(self):
        """Print grid search results summary (sorted by accuracy)."""
        print("\n" + "="*70)
        print("GRID SEARCH RESULTS SUMMARY")
        print("="*70)
        
        sorted_results = sorted(self.grid_results, key=lambda x: x['val_acc'], reverse=True)
        
        for i, res in enumerate(sorted_results, 1):
            p = res['params']
            print(f"{i:2d}. Val Acc: {res['val_acc']:.4f} | Units: {p['gru_units']:3d} | Dropout: {p['dropout']:.2f} | LR: {p['fine_tune_lr']:.0e} | Time: {res['time_s']:.1f}s")
        
        print("\n" + "="*70)
        print(f"BEST: Val Acc {sorted_results[0]['val_acc']:.4f}")
        print(f"PARAMS: {sorted_results[0]['params']}")
        print("="*70)
    
    def train_final_model(self, X_train, y_train, X_val, y_val, 
                         embedding_matrix, vocab_size, params=None):
        """Train final model with best parameters."""
        if params is None:
            params = self.best_params
        
        print(f"\n=== Training Final Model ===")
        print(f"Parameters: {params}")
        print(f"Epochs: Warmup={FINAL_WARMUP_EPOCHS}, Finetune={FINAL_FINETUNE_EPOCHS}")
        
        model = GRUModel(
            vocab_size, embedding_matrix,
            gru_units=params['gru_units'],
            dropout=params['dropout'],
            spatial_dropout=params['spatial_dropout'],
            dense_units=params['dense_units']
        )
        
        h1, h2 = model.train_freeze_thaw(
            X_train, y_train, X_val, y_val,
            FINAL_WARMUP_EPOCHS, FINAL_FINETUNE_EPOCHS,
            params.get('warmup_lr', 1e-3), params['fine_tune_lr'], self.batch_size
        )
        
        self.best_model = model
        
        
        return model, h1, h2
    
    def evaluate_and_report(self, model, X_val, y_val):
        """Generate evaluation report (print only)."""
        print("\n=== Final Evaluation ===")
        
        predictions = model.predict(X_val, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        
        acc = float(np.mean(y_pred == y_val))
        print(f"\nValidation accuracy: {acc:.4f}")
        

        
        return y_pred
    
    def run_full_pipeline(self, train_path=TRAIN_PATH, val_path=VAL_PATH):
        """Run the complete training pipeline (results printed, minimal file output)."""
        # Load data
        X_train, y_train, X_val, y_val, tokenizer = self.load_data(train_path, val_path)
        
        # Build embedding matrix
        print("\n=== Building Embedding Matrix ===")
        vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
        embedding_matrix = build_embedding_matrix(tokenizer.word_index)
        
        # Grid search
        best_params = self.run_grid_search(
            X_train, y_train, X_val, y_val, embedding_matrix, vocab_size
        )
        
        # Train final model
        final_model, h1, h2 = self.train_final_model(
            X_train, y_train, X_val, y_val, embedding_matrix, vocab_size, best_params
        )
        
        # Evaluate
        self.evaluate_and_report(final_model, X_val, y_val)
        
        print("\n=== PIPELINE COMPLETE ===")
        
        return final_model, best_params


def main():
    print("Bidirectional GRU Emotion Classification")
    trainer = GRUTrainer()
    model, params = trainer.run_full_pipeline()
    
    return model, params


if __name__ == "__main__":
    model, params = main()