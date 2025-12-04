"""
EmotionGRU model module for the Emotion Detection project.
Implements a Bidirectional GRU model as a tf.keras.Model subclass.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding,
    GRU,
    Dense,
    Dropout,
    Bidirectional,
    SpatialDropout1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .config import EMBED_DIM, NUM_CLASSES


class EmotionGRU(tf.keras.Model):
    """Bidirectional GRU model for emotion classification.
    
    This model extends tf.keras.Model and implements a bidirectional GRU
    architecture with GloVe embeddings and freeze-thaw training support.
    
    Architecture:
        - Embedding layer (with pre-trained GloVe weights)
        - SpatialDropout1D
        - Single Bidirectional GRU layer
        - Dense layer with ReLU
        - Dropout
        - Dense output layer with softmax
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_matrix: Pre-trained embedding matrix
        gru_units: Number of GRU units per layer (default: 128)
        dense_units: Number of units in hidden dense layer (default: 64)
        (single) number of GRU layers (single layer only)
        dropout: Dropout rate after dense layer (default: 0.5)
        spatial_dropout: Spatial dropout rate after embedding (default: 0.2)
        recurrent_dropout: Dropout rate within GRU (default: 0.0)
        num_classes: Number of output classes (default from config)
        embed_dim: Embedding dimension (default from config)
    """
    
    def __init__(self, vocab_size, embedding_matrix, 
                 gru_units=128, dense_units=64,
                 dropout=0.5, spatial_dropout=0.2, recurrent_dropout=0.0,
                 num_classes=NUM_CLASSES, embed_dim=EMBED_DIM, **kwargs):
        super(EmotionGRU, self).__init__(**kwargs)
        
        # Store hyperparameters
        self.vocab_size = vocab_size
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.dropout_rate = dropout
        self.spatial_dropout_rate = spatial_dropout
        self.recurrent_dropout_rate = recurrent_dropout
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Embedding layer with pre-trained weights
        self.embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            weights=[embedding_matrix],
            trainable=False,  # Start frozen
            mask_zero=True,
            name='embedding'
        )
        
        # Spatial dropout after embedding
        self.spatial_dropout = SpatialDropout1D(spatial_dropout, name='spatial_dropout')
        
        # Single Bidirectional GRU layer
        self.bidirectional_gru = Bidirectional(
            GRU(
                gru_units,
                return_sequences=False,
                recurrent_dropout=recurrent_dropout,
                name='gru'
            ),
            name='bidirectional_gru'
        )
        
        # Dense layers
        self.dense_hidden = Dense(dense_units, activation='relu', name='dense_hidden')
        self.dropout_layer = Dropout(dropout, name='dropout')
        self.dense_output = Dense(num_classes, activation='softmax', name='output')
    
    def call(self, inputs, training=None):
        """Forward pass of the model.
        
        Args:
            inputs: Input tensor of shape (batch_size, sequence_length)
            training: Boolean indicating training mode (for dropout)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Embedding + spatial dropout
        x = self.embedding(inputs)
        x = self.spatial_dropout(x, training=training)
        
        # Single GRU layer
        x = self.bidirectional_gru(x, training=training)
        
        # Dense layers
        x = self.dense_hidden(x)
        x = self.dropout_layer(x, training=training)
        return self.dense_output(x)
    
    def freeze_embeddings(self):
        """Freeze the embedding layer (disable training)."""
        self.embedding.trainable = False
        print("✓ Embeddings frozen")
    
    def unfreeze_embeddings(self):
        """Unfreeze the embedding layer (enable training)."""
        self.embedding.trainable = True
        print("✓ Embeddings unfrozen")
    
    def train_freeze_thaw(self, X_train, y_train, X_val, y_val,
                          warmup_epochs, finetune_epochs, 
                          warmup_lr, fine_tune_lr, batch_size):
        """Train model using freeze-thaw strategy.
        
        Phase 1 (Warmup): Train with frozen embeddings using warmup_lr
        Phase 2 (Finetune): Train with unfrozen embeddings using fine_tune_lr
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            warmup_epochs: Number of warmup epochs
            finetune_epochs: Number of finetuning epochs
            warmup_lr: Learning rate for warmup phase
            fine_tune_lr: Learning rate for finetuning phase
            batch_size: Batch size for training
            
        Returns:
            tuple: (history1, history2) - Training histories for both phases
        """
        # Phase 1: Warmup with frozen embeddings
        self.freeze_embeddings()
        self.compile(
            optimizer=Adam(learning_rate=warmup_lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Phase 1: Warmup training (lr={warmup_lr})...")
        history1 = self.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=warmup_epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Phase 2: Fine-tune with unfrozen embeddings
        self.unfreeze_embeddings()
        self.compile(
            optimizer=Adam(learning_rate=fine_tune_lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        print(f"Phase 2: Fine-tuning (lr={fine_tune_lr})...")
        history2 = self.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=finetune_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history1, history2
    
    def get_config(self):
        """Get model configuration for serialization."""
        config = super(EmotionGRU, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'gru_units': self.gru_units,
            'dense_units': self.dense_units,
            'dropout': self.dropout_rate,
            'spatial_dropout': self.spatial_dropout_rate,
            'recurrent_dropout': self.recurrent_dropout_rate,
            'num_classes': self.num_classes,
            'embed_dim': self.embed_dim,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)
    
    def summary_str(self):
        """Return a string summary of the model architecture."""
        return (
            f"EmotionGRU("
            f"gru_units={self.gru_units}, "
            f"layers=1, "
            f"dense={self.dense_units}, "
            f"dropout={self.dropout_rate}, "
            f"spatial={self.spatial_dropout_rate}, "
            f"recurrent={self.recurrent_dropout_rate})"
        )


def load_model(filepath):
    """Load a saved EmotionGRU model from disk.
    
    Args:
        filepath: Path to the saved model file
        
    Returns:
        Loaded Keras model
    """
    model = tf.keras.models.load_model(
        filepath,
        custom_objects={'EmotionGRU': EmotionGRU}
    )
    print(f"✓ Model loaded from {filepath}")
    return model
