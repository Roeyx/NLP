"""
Text preprocessing module for the Emotion Detection project.
Handles text cleaning, tokenization, vocabulary building, and sequence conversion.
"""

import re

import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .config import MAX_LEN


class TextPreprocessor:
    """Handles text preprocessing and tokenization for emotion classification."""
    
    def __init__(self):
        """Initialize the preprocessor with a TweetTokenizer."""
        self.tokenizer = TweetTokenizer(
            preserve_case=False,
            strip_handles=True,
            reduce_len=True
        )
        self.word_index = None
    
    @staticmethod
    def preprocess_text(text):
        """Clean and normalize text.
        
        Args:
            text: Input text string
            
        Returns:
            str: Cleaned and normalized text
        """
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_text(self, text_list):
        """Convert text list to token lists.
        
        Args:
            text_list: List of text strings
            
        Returns:
            list: List of token lists
        """
        tokenized = []
        for text in text_list:
            tokens = self.tokenizer.tokenize(str(text))
            tokenized.append(tokens)
        return tokenized
    
    def build_vocab(self, tokenized_texts):
        """Build vocabulary from tokenized texts.
        
        Args:
            tokenized_texts: List of token lists
            
        Returns:
            dict: Word to index mapping
        """
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
        """Convert token lists to integer sequences.
        
        Args:
            tokenized_texts: List of token lists
            word_index: Word to index mapping
            
        Returns:
            list: List of integer sequences
        """
        sequences = []
        for tokens in tokenized_texts:
            seq = [word_index.get(word, word_index['<UNK>']) for word in tokens]
            sequences.append(seq)
        return sequences
    
    def load_and_prep_data(self, filepath, word_index=None, is_train=True):
        """Load and preprocess data from CSV.
        
        Args:
            filepath: Path to CSV file
            word_index: Existing word index (for validation data)
            is_train: Whether this is training data (builds new vocab)
            
        Returns:
            tuple: (X, labels, word_index)
                - X: Padded sequences array
                - labels: Label array
                - word_index: Word to index mapping
        """
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
