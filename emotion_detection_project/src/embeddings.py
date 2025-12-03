"""
Embedding loader module for the Emotion Detection project.
Handles loading and caching of GloVe embeddings.
"""

import os

import numpy as np
from tqdm import tqdm

from .config import EMBED_DIM, GLOVE_FILE, EMBEDDING_CACHE_PATH
from .utils import timing_decorator


class EmbeddingLoader:
    """Loads and manages GloVe embeddings with caching support."""
    
    @staticmethod
    @timing_decorator
    def load_glove_matrix(word_index, embed_dim=EMBED_DIM, glove_path=GLOVE_FILE,
                          cache_path=EMBEDDING_CACHE_PATH):
        """Load GloVe embeddings from local file with caching support.
        
        Args:
            word_index: Dictionary mapping words to indices
            embed_dim: Embedding dimension (default from config)
            glove_path: Path to GloVe file (default from config)
            cache_path: Path to cache file (default from config)
            
        Returns:
            np.ndarray: Embedding matrix of shape (vocab_size, embed_dim)
        """
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
        embedding_matrix = EmbeddingLoader._build_embedding_matrix(
            word_index, embed_dim, glove_path
        )
        
        # Save to cache
        EmbeddingLoader._save_cache(embedding_matrix, word_index, cache_path)
        
        return embedding_matrix
    
    @staticmethod
    def _build_embedding_matrix(word_index, embed_dim, glove_path):
        """Build embedding matrix from GloVe file.
        
        Args:
            word_index: Dictionary mapping words to indices
            embed_dim: Embedding dimension
            glove_path: Path to GloVe file
            
        Returns:
            np.ndarray: Embedding matrix
        """
        print(f"Loading GloVe vectors from {glove_path}...")
        
        # Load embeddings from text file
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
                # Random initialization for OOV words
                embedding_matrix[i] = np.random.normal(scale=0.6, size=(embed_dim,))
                misses += 1
        
        coverage = hits / vocab_size
        print(f"Embedding Matrix Ready: {hits} hits, {misses} misses (coverage: {coverage:.2%})")
        
        return embedding_matrix
    
    @staticmethod
    def _save_cache(embedding_matrix, word_index, cache_path):
        """Save embedding matrix and word index to cache.
        
        Args:
            embedding_matrix: The embedding matrix to cache
            word_index: The word index dictionary
            cache_path: Path to save the cache
        """
        # Ensure cache directory exists
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        print(f"Saving embedding matrix to {cache_path}...")
        np.savez_compressed(
            cache_path,
            embedding_matrix=embedding_matrix,
            word_index=word_index
        )
        print("✓ Cache saved")
