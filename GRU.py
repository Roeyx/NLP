"""
Bidirectional GRU Model for Emotion Classification
Implements a GRU-based text classifier with GloVe embeddings and freeze-thaw training.
"""

import os
import sys
import time
from functools import wraps

# CRITICAL: Set LD_LIBRARY_PATH before importing TensorFlow
os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('CONDA_PREFIX', '')}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import tensorflow as tf
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt


def timing_decorator(func):
    """Decorator to measure execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        if elapsed_time < 60:
            print(f"⏱️  {func.__name__} completed in {elapsed_time:.2f} seconds")
        else:
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            print(f"⏱️  {func.__name__} completed in {minutes}m {seconds:.2f}s")
        
        return result
    return wrapper


# Configuration Constants
MAX_LEN = 50
EMBED_DIM = 200
GLOVE_FILE = "glove.twitter.27B.200d.txt"
GRID_WARMUP_EPOCHS = 2
GRID_FINETUNE_EPOCHS = 4
FINAL_WARMUP_EPOCHS = 5
FINAL_FINETUNE_EPOCHS = 15
TRAIN_PATH = "train.csv"
VAL_PATH = "validation.csv"

LABEL_MAP = {
    0: 'sadness', 1: 'joy', 2: 'love',
    3: 'anger', 4: 'fear', 5: 'surprise'
}

PARAM_GRID = {
    'gru_units': [128, 256],           
    'dropout': [0.3, 0.5],             
    'fine_tune_lr': [1e-4, 5e-5],      
    'spatial_dropout': [0.2],          
    'dense_units': [64, 128],          
}



class TextPreprocessor:
    """Handles text preprocessing and tokenization."""
    
    def __init__(self):
        self.tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        self.word_index = None
    
    @staticmethod
    def preprocess_text(text):
        """Clean and normalize text."""
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_text(self, text_list):
        """Convert text list to token lists."""
        tokenized = []
        for text in text_list:
            tokens = self.tokenizer.tokenize(str(text))
            tokenized.append(tokens)
        return tokenized
    
    def build_vocab(self, tokenized_texts):
        """Build vocabulary from tokenized texts."""
        word_index = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for tokens in tokenized_texts:
            for word in tokens:
                if word not in word_index:
                    word_index[word] = idx
                    idx += 1
        self.word_index = word_index
        return word_index
    
    def text_to_sequences(self, tokenized_texts, word_index):
        """Convert token lists to integer sequences."""
        sequences = []
        for tokens in tokenized_texts:
            seq = [word_index.get(word, word_index['<UNK>']) for word in tokens]
            sequences.append(seq)
        return sequences
    
    def load_and_prep_data(self, filepath, word_index=None, is_train=True):
        """Load and preprocess data from CSV."""
        print(f"Loading {filepath}...")
        df = pd.read_csv(filepath)
        
        texts = df['text'].apply(self.preprocess_text).values
        labels = df['label'].values
        
        tokenized = self.tokenize_text(texts)
        
        if is_train:
            word_vocab = self.build_vocab(tokenized)
        else:
            word_vocab = word_index
        
        sequences = self.text_to_sequences(tokenized, word_vocab)
        X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        
        return X, labels, word_vocab


class EmbeddingLoader:
    """Loads and manages GloVe embeddings."""
    
    @staticmethod
    @timing_decorator
    def load_glove_matrix(word_index, embed_dim=EMBED_DIM, glove_path=GLOVE_FILE, 
                         cache_path="embedding_matrix.npz"):
        """Load GloVe embeddings from local file with caching support."""
        
        # Try to load from cache first
        if os.path.exists(cache_path):
            print(f"Loading cached embedding matrix from {cache_path}...")
            try:
                data = np.load(cache_path, allow_pickle=True)
                cached_word_index = data['word_index'].item()
                
                # Check if vocabulary matches
                if cached_word_index == word_index:
                    embedding_matrix = data['embedding_matrix']
                    print(f"✓ Loaded cached embedding matrix: shape {embedding_matrix.shape}")
                    return embedding_matrix
                else:
                    print("⚠️  Vocabulary changed, rebuilding embedding matrix...")
            except Exception as e:
                print(f"⚠️  Failed to load cache ({e}), rebuilding...")
        
        # Build embedding matrix from scratch
        print(f"Loading GloVe vectors from {glove_path}...")
        
        # Load embeddings from text file (fast numpy method)
        print("Reading file...")
        embeddings_index = {}
        data = []
        words = []
        
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Parsing GloVe"):
                parts = line.rstrip().split(' ')
                words.append(parts[0])
                data.append(list(map(float, parts[1:])))
        
        # Convert to numpy array for faster lookup
        data_array = np.array(data, dtype='float32')
        embeddings_index = dict(zip(words, data_array))
        
        print(f"Found {len(embeddings_index)} word vectors")
        
        # Build embedding matrix
        vocab_size = len(word_index)
        embedding_matrix = np.zeros((vocab_size, embed_dim), dtype="float32")
        
        hits = 0
        misses = 0
        
        for word, i in tqdm(word_index.items(), desc="Building Embedding Matrix"):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                embedding_matrix[i] = np.random.normal(scale=0.6, size=(embed_dim,))
                misses += 1
        
        print(f"Embedding Matrix Ready: {hits} hits, {misses} misses (coverage: {hits/vocab_size:.2%})")
        
        # Save to cache
        print(f"Saving embedding matrix to {cache_path}...")
        np.savez_compressed(cache_path, 
                           embedding_matrix=embedding_matrix,
                           word_index=word_index)
        print("✓ Cache saved")
        
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
                output_dim=EMBED_DIM,
                weights=[self.embedding_matrix],
                trainable=False,
                mask_zero=True
            ),
            SpatialDropout1D(self.spatial_dropout),
            Bidirectional(GRU(
                self.gru_units,
                return_sequences=False,
                recurrent_activation='sigmoid',
                reset_after=False
            )),
            Dense(self.dense_units, activation='relu'),
            Dropout(self.dropout),
            Dense(len(LABEL_MAP), activation='softmax')
        ])
        
        self.model = model
        return model
    
    def train_freeze_thaw(self, X_train, y_train, X_val, y_val, 
                         warmup_epochs, finetune_epochs, fine_tune_lr, batch_size):
        """Train model using freeze-thaw strategy."""
        if self.model is None:
            self.build_model()
        
        # Phase 1: Warmup with frozen embeddings
        self.model.layers[0].trainable = False
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history1 = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=warmup_epochs,
            batch_size=batch_size,
            verbose=0
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
            verbose=0
        )
        
        return history1, history2
    
    def evaluate(self, X_val, y_val):
        """Evaluate model on validation data."""
        loss, acc = self.model.evaluate(X_val, y_val, verbose=0)
        return loss, acc
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def save(self, filepath="best_model.keras"):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Build and train a model first.")
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")
    
    @staticmethod
    def load(filepath="best_model.keras"):
        """Load a saved model from disk."""
        model = tf.keras.models.load_model(filepath)
        print(f"✓ Model loaded from {filepath}")
        return model


class GRUTrainer:
    """Manages training pipeline including grid search."""
    
    def __init__(self, param_grid=PARAM_GRID):
        self.param_grid = param_grid
        self.preprocessor = TextPreprocessor()
        self.best_params = None
        self.best_model = None
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        self.has_gpu = len(gpus) > 0
        self.batch_size = 64 if self.has_gpu else 32
        
        if self.has_gpu:
            print(f"✅ GPU Detected: {len(gpus)} device(s)")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Set BATCH_SIZE = {self.batch_size} (GPU-optimized)")
        else:
            print(f"⚠️ No GPU detected")
            print(f"✓ Set BATCH_SIZE = {self.batch_size} (CPU-optimized)")
    
    def load_data(self, train_path=TRAIN_PATH, val_path=VAL_PATH):
        """Load and preprocess training and validation data."""
        print("\n=== Loading Data ===")
        X_train, y_train, word_index = self.preprocessor.load_and_prep_data(
            train_path, is_train=True
        )
        X_val, y_val, _ = self.preprocessor.load_and_prep_data(
            val_path, word_index=word_index, is_train=False
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Vocabulary size: {len(word_index)}")
        
        return X_train, y_train, X_val, y_val, word_index
    
    def run_grid_search(self, X_train, y_train, X_val, y_val, embedding_matrix, vocab_size):
        """Run grid search to find best hyperparameters."""
        grid = list(ParameterGrid(self.param_grid))
        total = len(grid)
        
        print(f"\n=== Grid Search: {total} combinations ===")
        print("⏳ Note: First iteration will be slow due to GPU warmup/compilation...")
        
        best_acc = 0.0
        best_params = None
        
        for i, params in enumerate(grid, 1):
            print(f"\n[{i}/{total}] Testing: {params}...")
            iter_start = time.time()
            
            model = GRUModel(
                vocab_size, embedding_matrix,
                gru_units=params['gru_units'],
                dropout=params['dropout'],
                spatial_dropout=params['spatial_dropout'],
                dense_units=params['dense_units']
            )
            
            print("  → Training warmup phase...")
            model.train_freeze_thaw(
                X_train, y_train, X_val, y_val,
                GRID_WARMUP_EPOCHS, GRID_FINETUNE_EPOCHS,
                params['fine_tune_lr'], self.batch_size
            )
            
            print("  → Evaluating...")
            _, val_acc = model.evaluate(X_val, y_val)
            iter_time = time.time() - iter_start
            print(f"  ✓ Val Acc: {val_acc:.4f} (took {iter_time:.1f}s)")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = params
        
        print("\n" + "="*50)
        print(f"BEST RESULT: {best_acc:.4f}")
        print(f"BEST PARAMS: {best_params}")
        print("="*50)
        
        self.best_params = best_params
        
        # Save best params to file
        import json
        with open('best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        print("✓ Best params saved to best_params.json")
        
        return best_params
    
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
            params['fine_tune_lr'], self.batch_size
        )
        
        self.best_model = model
        
        # Save the final model
        model.save("final_model.keras")
        print("✓ Final model saved to final_model.keras")
        
        return model, h1, h2
    
    def evaluate_and_report(self, model, X_val, y_val):
        """Generate evaluation report and plot."""
        print("\n=== Final Evaluation ===")
        
        predictions = model.predict(X_val)
        y_pred = np.argmax(predictions, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=list(LABEL_MAP.values())))
        
        return y_pred
    
    def plot_training_history(self, h1, h2, params, save_path='best_model_history.png'):
        """Plot and save training history."""
        print("\n=== Plotting Results ===")
        
        acc = h1.history['accuracy'] + h2.history['accuracy']
        val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
        
        plt.figure(figsize=(10, 5))
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.axvline(x=len(h1.history['accuracy']), color='r', linestyle='--', 
                   label='Fine-Tuning Start')
        plt.title(f'Final Model Training History\nParams: {params}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved plot to {save_path}")
    
    def run_full_pipeline(self, train_path=TRAIN_PATH, val_path=VAL_PATH):
        """Run the complete training pipeline."""
        # Load data
        X_train, y_train, X_val, y_val, word_index = self.load_data(train_path, val_path)
        
        # Load embeddings
        print("\n=== Building Embedding Matrix ===")
        vocab_size = len(word_index)
        embedding_matrix = EmbeddingLoader.load_glove_matrix(word_index)
        
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
        
        # Plot
        self.plot_training_history(h1, h2, best_params)
        
        # Save vocabulary for future inference
        import pickle
        with open('word_index.pkl', 'wb') as f:
            pickle.dump(word_index, f)
        print("✓ Vocabulary saved to word_index.pkl")
        
        print("\n=== DONE ===")
        print("\nSaved files:")
        print("  - final_model.keras (trained model)")
        print("  - best_params.json (hyperparameters)")
        print("  - word_index.pkl (vocabulary)")
        print("  - embedding_matrix.npz (cached embeddings)")
        print("  - best_model_history.png (training plot)")
        
        return final_model, best_params


def main():
    """Main entry point."""
    print("="*60)
    print("Bidirectional GRU Emotion Classification")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python: {sys.version.split()[0]}")
    
    trainer = GRUTrainer()
    model, params = trainer.run_full_pipeline()
    
    return model, params


if __name__ == "__main__":
    model, params = main()