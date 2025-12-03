"""
Trainer module for the Emotion Detection project.
Manages the training pipeline including data loading, grid search, and evaluation.
"""

import json
import pickle
import time
import os

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid

from .config import (
    TRAIN_PATH,
    VAL_PATH,
    PARAM_GRID,
    BATCH_SIZE,
    GRID_WARMUP_EPOCHS,
    GRID_FINETUNE_EPOCHS,
    FINAL_WARMUP_EPOCHS,
    FINAL_FINETUNE_EPOCHS,
    LABEL_MAP,
    FINAL_MODEL_PATH,
    GRID_RESULTS_PATH,
    BEST_PARAMS_PATH,
    WORD_INDEX_PATH,
    CLASSIFICATION_REPORT_PATH,
    RESULTS_DIR,
)
from .preprocessing import TextPreprocessor
from .embeddings import EmbeddingLoader
from .model import EmotionGRU
from .utils import setup_gpu, timing_decorator


class GRUTrainer:
    """Manages training pipeline including grid search for EmotionGRU model.
    
    This class orchestrates the complete training workflow:
    1. Data loading and preprocessing
    2. Embedding matrix construction
    3. Hyperparameter grid search
    4. Final model training
    5. Evaluation and reporting
    6. Model and artifact saving
    
    Args:
        param_grid: Dictionary of hyperparameter options for grid search
    """
    
    def __init__(self, param_grid=PARAM_GRID):
        """Initialize the trainer with parameter grid and GPU setup."""
        self.param_grid = param_grid
        self.preprocessor = TextPreprocessor()
        self.best_params = None
        self.best_model = None
        self.batch_size = BATCH_SIZE
        
        # Setup GPU
        self.has_gpu, _ = setup_gpu()
    
    def load_data(self, train_path=TRAIN_PATH, val_path=VAL_PATH):
        """Load and preprocess training and validation data.
        
        Args:
            train_path: Path to training CSV file
            val_path: Path to validation CSV file
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val, word_index)
        """
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
    
    def _create_model(self, vocab_size, embedding_matrix, params):
        """Create a model instance with given parameters.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_matrix: Pre-trained embedding weights
            params: Dictionary of hyperparameters
            
        Returns:
            EmotionGRU: Configured model instance
        """
        return EmotionGRU(
            vocab_size=vocab_size,
            embedding_matrix=embedding_matrix,
            gru_units=params.get('gru_units', 128),
            dense_units=params.get('dense_units', 64),
            dropout=params.get('dropout', 0.5),
            spatial_dropout=params.get('spatial_dropout', 0.2),
            recurrent_dropout=params.get('recurrent_dropout', 0.0),
        )
    
    @timing_decorator
    def run_grid_search(self, X_train, y_train, X_val, y_val, 
                        embedding_matrix, vocab_size):
        """Run grid search to find best hyperparameters.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            embedding_matrix: Pre-trained embedding matrix
            vocab_size: Size of vocabulary
            
        Returns:
            dict: Best hyperparameters found
        """
        grid = list(ParameterGrid(self.param_grid))
        total = len(grid)
        
        print(f"\n=== Grid Search: {total} combinations ===")
        print(f"Batch size: {self.batch_size}")
        print("⏳ Note: First iteration will be slow due to GPU warmup/compilation...")
        
        best_acc = 0.0
        best_params = None
        
        for i, params in enumerate(grid, 1):
            print(f"\n[{i}/{total}] Testing: {params}...")
            iter_start = time.time()
            
            try:
                model = self._create_model(vocab_size, embedding_matrix, params)
                
                warmup_lr = params.get('warmup_lr', 1e-3)
                fine_tune_lr = params.get('fine_tune_lr', 1e-4)
                
                _, h2 = model.train_freeze_thaw(
                    X_train, y_train, X_val, y_val,
                    GRID_WARMUP_EPOCHS, GRID_FINETUNE_EPOCHS,
                    warmup_lr, fine_tune_lr, self.batch_size
                )
                
                val_acc = max(h2.history['val_accuracy'])
                iter_time = time.time() - iter_start
                
                print(f"  ✓ Val Acc: {val_acc:.4f} (took {iter_time:.1f}s)")
                
                # Append result to JSON file
                self._append_result_to_json(params, val_acc, iter_time)
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_params = params.copy()
                    self._save_best_params(best_params)
                    print(f"  *** New best! ***")
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        print("\n" + "=" * 50)
        print(f"BEST RESULT: {best_acc:.4f}")
        print(f"BEST PARAMS: {best_params}")
        print("=" * 50)
        
        if best_params is None:
            raise RuntimeError(
                "Grid search failed: no parameter configurations completed successfully. "
                "Check the errors above for details."
            )
        
        self.best_params = best_params
        return best_params
    
    @timing_decorator
    def train_final_model(self, X_train, y_train, X_val, y_val,
                          embedding_matrix, vocab_size, params=None):
        """Train final model with best parameters.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            embedding_matrix: Pre-trained embedding matrix
            vocab_size: Size of vocabulary
            params: Hyperparameters to use (defaults to best_params)
            
        Returns:
            tuple: (model, history1, history2)
        """
        if params is None:
            params = self.best_params
        
        print(f"\n=== Training Final Model ===")
        print(f"Parameters: {params}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: Warmup={FINAL_WARMUP_EPOCHS}, Finetune={FINAL_FINETUNE_EPOCHS}")
        
        model = self._create_model(vocab_size, embedding_matrix, params)
        
        warmup_lr = params.get('warmup_lr', 1e-3)
        fine_tune_lr = params.get('fine_tune_lr', 1e-4)
        
        h1, h2 = model.train_freeze_thaw(
            X_train, y_train, X_val, y_val,
            FINAL_WARMUP_EPOCHS, FINAL_FINETUNE_EPOCHS,
            warmup_lr, fine_tune_lr, self.batch_size
        )
        
        self.best_model = model
        
        # Save the final model
        model.save(FINAL_MODEL_PATH)
        print(f"✓ Final model saved to {FINAL_MODEL_PATH}")
        
        return model, h1, h2
    
    def evaluate_and_report(self, model, X_val, y_val):
        """Generate evaluation report.
        
        Args:
            model: Trained EmotionGRU model
            X_val: Validation sequences
            y_val: Validation labels
            
        Returns:
            np.ndarray: Predicted labels
        """
        print("\n=== Final Evaluation ===")
        
        predictions = model.predict(X_val, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        
        # Generate classification report
        report = classification_report(
            y_val, y_pred,
            target_names=list(LABEL_MAP.values())
        )
        
        print("\nClassification Report:")
        print(report)
        
        # Save report to file
        self._save_classification_report(report)
        
        return y_pred
    
    def run_full_pipeline(self, train_path=TRAIN_PATH, val_path=VAL_PATH):
        """Run the complete training pipeline.
        
        Args:
            train_path: Path to training CSV file
            val_path: Path to validation CSV file
            
        Returns:
            tuple: (final_model, best_params)
        """
        # Load data
        X_train, y_train, X_val, y_val, word_index = self.load_data(
            train_path, val_path
        )
        
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
        
        # Save vocabulary
        self._save_word_index(word_index)
        
        # Print summary
        self._print_summary()
        
        return final_model, best_params
    
    def _append_result_to_json(self, params, val_accuracy, time):
        """Append a grid search result to the JSON file.
        
        Args:
            params: Dictionary of hyperparameters
            val_accuracy: Validation accuracy achieved
            time: Time taken for this iteration
        """
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Load existing results or create new list
        if os.path.exists(GRID_RESULTS_PATH):
            with open(GRID_RESULTS_PATH, 'r') as f:
                results = json.load(f)
        else:
            results = []
        
        # Append new result
        results.append({
            'params': params,
            'val_accuracy': val_accuracy,
            'time': time
        })
        
        # Save back to file
        with open(GRID_RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  → Appended result to {GRID_RESULTS_PATH}")
    
    def _save_best_params(self, params):
        """Save best parameters to JSON file."""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        with open(BEST_PARAMS_PATH, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"  ✓ Best params saved to {BEST_PARAMS_PATH}")

    def _save_word_index(self, word_index):
        """Save word index to pickle file."""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        with open(WORD_INDEX_PATH, 'wb') as f:
            pickle.dump(word_index, f)
        print(f"✓ Vocabulary saved to {WORD_INDEX_PATH}")
    
    def _save_classification_report(self, report):
        """Save classification report to text file."""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        with open(CLASSIFICATION_REPORT_PATH, 'w') as f:
            f.write(report)
        print(f"✓ Classification report saved to {CLASSIFICATION_REPORT_PATH}")
    
    def _print_summary(self):
        """Print summary of saved files."""
        print("\n=== DONE ===")
        print("\nSaved files:")
        print(f"  - {FINAL_MODEL_PATH} (trained model)")
        print(f"  - {BEST_PARAMS_PATH} (best hyperparameters)")
        print(f"  - {GRID_RESULTS_PATH} (all grid search results)")
        print(f"  - {WORD_INDEX_PATH} (vocabulary)")
        print(f"  - {CLASSIFICATION_REPORT_PATH} (evaluation report)")
