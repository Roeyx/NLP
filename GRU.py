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

# The dataset folder contains 'glove.twitter.27B.100d.txt'
GLOVE_FILE = os.path.join(path, "glove.twitter.27B.100d.txt")

MAX_WORDS = 30000
MAX_LEN = 100 
EMBED_DIM = 100  

# Training epochs parameters for fine-tuning phase
GRID_WARMUP_EPOCHS = 2
GRID_FINETUNE_EPOCHS = 3
FINAL_WARMUP_EPOCHS = 5
FINAL_FINETUNE_EPOCHS = 15

# Baseline config to match LSTM.py settings
BASELINE_CONFIG = {
    'batch_size': 32, 
    'epochs': 20,                       
}

# LSTM parity constants — exact units and dropout ranges used in LSTM.py
LSTM_PARITY_UNITS = [16, 32, 64, 128, 256]
LSTM_PARITY_DROPOUT = list(np.arange(0.05, 0.55, 0.05))

LABEL_MAP = {
    0: 'sadness', 1: 'joy', 2: 'love',
    3: 'anger', 4: 'fear', 5: 'surprise'
}

# Phase 1 Grid: LSTM parity — 5 units × 10 dropout = 50 combos
PHASE1_PARAM_GRID = {
    # use the LSTM parity units/dropout by default — Phase 1 must match LSTM's grid
    'gru_units': LSTM_PARITY_UNITS,
    'dropout': LSTM_PARITY_DROPOUT,
    'spatial_dropout': [0.2],
    'dense_units': [64],
}

# Phase 2 Grid for freeze-thaw optimization (engineering phase)
PHASE2_PARAM_GRID = {
    'gru_units': [128, 256],
    'dropout': [0.2, 0.3],
    'fine_tune_lr': [1e-3, 5e-4, 1e-4, 5e-5],
    'warmup_lr': [1e-3],
    'spatial_dropout': [0.2],
    'dense_units': [64],
}

# Phase 2 uses larger batch size for optimization
PHASE2_BATCH_SIZE = 128

# Output formatting
SEPARATOR = '=' * 70


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
    
    def build_model(self, use_extra_dense: bool = True, trainable_embeddings: bool = True):
        """Build the bidirectional GRU model architecture.
        Args:
            use_extra_dense (bool): Whether to include the intermediate dense + dropout layers (optimized GRU).
            trainable_embeddings (bool): Whether to allow training of embedding layer.
        """
        layers = [
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_matrix.shape[1],
                weights=[self.embedding_matrix],
                input_length=MAX_LEN,
                trainable=trainable_embeddings,
                mask_zero=True
            ),
            SpatialDropout1D(self.spatial_dropout),
            Bidirectional(GRU(self.gru_units, return_sequences=False)),
        ]
        # optional extra dense + dropout for fine tuning
        if use_extra_dense:
            layers += [
                Dense(self.dense_units, activation='relu'),
                Dropout(self.dropout),
            ]

        # output layer
        layers += [Dense(len(LABEL_MAP), activation='softmax')]

        model = Sequential(layers)
        
        self.model = model
        return model
    
    def train_freeze_thaw(self, X_train, y_train, X_val, y_val, 
                         warmup_epochs, finetune_epochs, warmup_lr, fine_tune_lr, batch_size):
        """Train model using freeze-thaw strategy for fine tuning."""
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

    def train_baseline(self, X_train, y_train, X_val, y_val, batch_size, epochs):
        """Train model in single-phase baseline mode 

        Uses standard Adam optimizer and callbacks:
            - EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            - ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)

        Returns the Keras History object for the training run.
        """
        if self.model is None:
            # Default to the 'use_extra_dense' and embedding options used when building
            self.build_model()

        # Compile with standard Adam optimizer (default lr=0.001)
        self.model.compile(
            optimizer=Adam(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=0)
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history
    
    def predict(self, X, **kwargs):
        """Make predictions."""
        return self.model.predict(X, **kwargs)


class GRUTrainer:
    """Manages training pipeline including grid search and final training."""
    
    def __init__(self, param_grid=PHASE1_PARAM_GRID):
        self.param_grid = param_grid
        self.best_params = None
        self.best_model = None
        self.grid_results = []
        # Ensure baseline batch/epochs is available to all training phases
        self.batch_size = BASELINE_CONFIG['batch_size']
        self.epochs = BASELINE_CONFIG['epochs']
            
    
    def load_data(self, train_path=TRAIN_PATH, val_path=VAL_PATH):
        """Load and preprocess data """
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
        
        # Encode labels with LabelEncoder 
        le = LabelEncoder()
        y_train = le.fit_transform(train_df['label'])
        y_val = le.transform(val_df['label'])
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Tokenizer vocab size: {min(MAX_WORDS, len(tokenizer.word_index) + 1)}")
        
        return X_train, y_train, X_val, y_val, tokenizer
    
    def run_phase1_grid_search(self, X_train, y_train, X_val, y_val, embedding_matrix, vocab_size):
        """Phase 1: Run grid search under LSTM baseline conditions for comparison."""
        # Build Phase 1 grid (LSTM parity enforcement)
        grid = list(ParameterGrid(PHASE1_PARAM_GRID))
        total = len(grid)
        
        print(f"\n{SEPARATOR}")
        print(f"PHASE 1: BASELINE GRID SEARCH (LSTM Parity)")
        print(f"{SEPARATOR}")
        print(f"Grid size: {total} combinations")
        print(f"Units: {LSTM_PARITY_UNITS}")
        print(f"Dropout: [0.05, 0.10, ..., 0.50] (10 values)")
        print(f"Batch size: {BASELINE_CONFIG['batch_size']}, Epochs: {BASELINE_CONFIG['epochs']}")
        print(f"Embeddings: Static (trainable=False), Architecture: No extra dense layer")
        print("⏳ First iteration may be slow (GPU warmup)...\n")
        
        best_acc = 0.0
        best_params = None
        self.grid_results = []
        
        for i, params in enumerate(grid, 1):
            print(f"[{i}/{total}] Units={params['gru_units']}, Dropout={params['dropout']:.2f}")
            start = time.time()
            
            model = GRUModel(
                vocab_size, embedding_matrix,
                gru_units=params['gru_units'],
                dropout=params['dropout'],
                spatial_dropout=params['spatial_dropout'],
                dense_units=params['dense_units']
            )
            # Phase 1: Build model under baseline (static embeddings, no extra dense)
            model.build_model(use_extra_dense=False,
                              trainable_embeddings=False)

            # Train using the fair baseline procedure
            history = model.train_baseline(
                X_train, y_train, X_val, y_val,
                BASELINE_CONFIG['batch_size'], BASELINE_CONFIG['epochs']
            )

            # Get best validation accuracy from history
            val_accs = history.history.get('val_accuracy') or history.history.get('val_acc')
            best_val_acc = max(val_accs)
            elapsed = time.time() - start
            
            result = {
                'params': params,
                'val_acc': float(best_val_acc),
                'time_s': float(elapsed)
            }
            self.grid_results.append(result)
            
            print(f"  → Val Acc: {best_val_acc:.4f} (took {elapsed:.1f}s)")
            
            if best_val_acc > best_acc:
                best_acc = best_val_acc
                best_params = params
        
        self.best_params = best_params
        self._print_grid_summary(phase_name="Phase 1")
        
        return best_params
    
    def run_phase2_grid_search(self, X_train, y_train, X_val, y_val, embedding_matrix, vocab_size, phase1_best_units, phase1_best_dropout):
        """Phase 2: Run freeze-thaw grid search with engineering optimizations."""
        grid = list(ParameterGrid(PHASE2_PARAM_GRID))
        total = len(grid)
        
        print(f"\n{SEPARATOR}")
        print(f"PHASE 2: FREEZE-THAW OPTIMIZATION GRID SEARCH")
        print(f"{SEPARATOR}")
        print(f"Grid size: {total} combinations")
        print(f"Units: {PHASE2_PARAM_GRID['gru_units']}")
        print(f"Dropout: {PHASE2_PARAM_GRID['dropout']}")
        print(f"Learning rates: {PHASE2_PARAM_GRID['fine_tune_lr']}")
        print(f"Batch size: {PHASE2_BATCH_SIZE} (larger for optimization)")
        print(f"Embeddings: Trainable (freeze-thaw), Architecture: With extra dense layer")
        print(f"Using Phase 1 best baseline: Units={phase1_best_units}, Dropout={phase1_best_dropout:.2f}")
        print("⏳ Running freeze-thaw training...\n")
        
        best_acc = 0.0
        best_params = None
        phase2_results = []
        
        for i, params in enumerate(grid, 1):
            print(f"[{i}/{total}] Units={params['gru_units']}, Dropout={params['dropout']:.2f}, LR={params['fine_tune_lr']:.0e}")
            start = time.time()
            
            model = GRUModel(
                vocab_size, embedding_matrix,
                gru_units=params['gru_units'],
                dropout=params['dropout'],
                spatial_dropout=params['spatial_dropout'],
                dense_units=params['dense_units']
            )
            # Phase 2: Build with optimizations
            model.build_model(use_extra_dense=True, trainable_embeddings=True)
            
            # Train using freeze-thaw
            h1, h2 = model.train_freeze_thaw(
                X_train, y_train, X_val, y_val,
                GRID_WARMUP_EPOCHS, GRID_FINETUNE_EPOCHS,
                params.get('warmup_lr', 1e-3), params['fine_tune_lr'], PHASE2_BATCH_SIZE
            )
            
            # Get best validation accuracy from finetune history
            val_accs = h2.history.get('val_accuracy') or h2.history.get('val_acc')
            best_val_acc = max(val_accs)
            elapsed = time.time() - start
            
            result = {
                'params': params,
                'val_acc': float(best_val_acc),
                'time_s': float(elapsed)
            }
            phase2_results.append(result)
            
            print(f"  → Val Acc: {best_val_acc:.4f} (took {elapsed:.1f}s)")
            
            if best_val_acc > best_acc:
                best_acc = best_val_acc
                best_params = params
        
        # Store phase 2 results separately
        self.phase2_results = phase2_results
        self.phase2_best_params = best_params
        
        # Print Phase 2 summary
        self.grid_results = phase2_results  # Temporarily set for printing
        self._print_grid_summary(phase_name="Phase 2")
        
        return best_params
    
    def _print_grid_summary(self, phase_name="Phase 1"):
        """Print grid search results summary (sorted by accuracy)."""
        print(f"\n{SEPARATOR}")
        print(f"{phase_name.upper()} GRID SEARCH RESULTS SUMMARY")
        print(SEPARATOR)
        
        sorted_results = sorted(self.grid_results, key=lambda x: x['val_acc'], reverse=True)
        
        for i, res in enumerate(sorted_results, 1):
            p = res['params']
            # Handle optional keys gracefully
            lr_str = f"| LR: {p['fine_tune_lr']:.0e}" if 'fine_tune_lr' in p else ""
            print(f"{i:2d}. Val Acc: {res['val_acc']:.4f} | Units: {p['gru_units']:3d} | Dropout: {p['dropout']:.2f} {lr_str} | Time: {res['time_s']:.1f}s")
        
        print(f"\n{SEPARATOR}")
        print(f"BEST: Val Acc {sorted_results[0]['val_acc']:.4f}")
        print(f"PARAMS: {sorted_results[0]['params']}")
        print(SEPARATOR)
    
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
        # Build final model with engineering optimizations (trainable embeddings + extra dense)
        model.build_model(use_extra_dense=True, trainable_embeddings=True)

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
        
        return y_pred, acc
    
    def run_full_pipeline(self, train_path=TRAIN_PATH, val_path=VAL_PATH):
        """Run the complete two-phase training pipeline."""
        # Load data
        X_train, y_train, X_val, y_val, tokenizer = self.load_data(train_path, val_path)
        
        # Build embedding matrix
        print("\n=== Building Embedding Matrix ===")
        vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
        embedding_matrix = build_embedding_matrix(tokenizer.word_index)
        
        # ========== PHASE 1: BASELINE GRID SEARCH (LSTM Parity) ==========
        phase1_best_params = self.run_phase1_grid_search(
            X_train, y_train, X_val, y_val, embedding_matrix, vocab_size
        )
        
        # Compute and display Phase 1 best result
        phase1_best_val = None
        if self.grid_results:
            phase1_best_val = max(r['val_acc'] for r in self.grid_results)
            print(f"\n{SEPARATOR}")
            print(f"✅ PHASE 1 COMPLETE: Best Baseline Accuracy = {phase1_best_val:.4f}")
            print(f"   Best hyperparameters: Units={phase1_best_params['gru_units']}, Dropout={phase1_best_params['dropout']:.2f}")
            print(SEPARATOR)
        
        # ========== PHASE 2: FREEZE-THAW OPTIMIZATION GRID SEARCH ==========
        phase2_best_params = self.run_phase2_grid_search(
            X_train, y_train, X_val, y_val, embedding_matrix, vocab_size,
            phase1_best_params['gru_units'], phase1_best_params['dropout']
        )
        
        # Compute and display Phase 2 best result
        phase2_best_val = None
        if hasattr(self, 'phase2_results') and self.phase2_results:
            phase2_best_val = max(r['val_acc'] for r in self.phase2_results)
            print(f"\n{SEPARATOR}")
            print(f"✅ PHASE 2 COMPLETE: Best Optimized Accuracy = {phase2_best_val:.4f}")
            print(f"   Best hyperparameters: Units={phase2_best_params['gru_units']}, Dropout={phase2_best_params['dropout']:.2f}, LR={phase2_best_params['fine_tune_lr']:.0e}")
            print(SEPARATOR)
        
        # ========== FINAL TRAINING with Phase 2 best params ==========
        print(f"\n{SEPARATOR}")
        print(f"FINAL TRAINING: Using Phase 2 best parameters")
        print(SEPARATOR)
        final_model, h1, h2 = self.train_final_model(
            X_train, y_train, X_val, y_val, embedding_matrix, vocab_size, phase2_best_params
        )
        
        # Final evaluation
        y_pred, final_acc = self.evaluate_and_report(final_model, X_val, y_val)
        
        print(f"\n{SEPARATOR}")
        print(f"🎯 FINAL MODEL ACCURACY: {final_acc:.4f}")
        print(f"   Phase 1 (Baseline): {phase1_best_val:.4f}")
        print(f"   Phase 2 (Optimized Grid): {phase2_best_val:.4f}")
        print(f"   Final (Extended Training): {final_acc:.4f}")
        print(SEPARATOR)
        print("\n=== PIPELINE COMPLETE ===")
        
        return final_model, {'phase1': phase1_best_params, 'phase2': phase2_best_params}


def main():
    print("Bidirectional GRU Emotion Classification")
    trainer = GRUTrainer()
    model, params = trainer.run_full_pipeline()
    
    return model, params


if __name__ == "__main__":
    model, params = main()