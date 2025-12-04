import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
import kagglehub
import os

# ==========================================
# 1. Setup & Configuration
# ==========================================
# Configuration Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
VAL_PATH = os.path.join(BASE_DIR, "validation.csv")


# Hyperparameters
# The dataset has 15,214 unique words therefore we chose larger MAX_WORDS hyperparameter
MAX_WORDS = 30000  # Max vocabulary size
MAX_LEN = 100  # Max sequence length
EMBED_DIM = 100  # Embedding dimension
EPOCHS = 20
BATCH_SIZE = 32

# Download latest version from Kaggle
path = kagglehub.dataset_download("bertcarremans/glovetwitter27b100dtxt")
print("Path to dataset files:", path)

# The dataset folder contains 'glove.twitter.27B.100d.txt'
GLOVE_PATH = os.path.join(path, "glove.twitter.27B.100d.txt")
print("Using GloVe file:", GLOVE_PATH)



# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
def preprocess_text(text):
    text = str(text).lower() #change text to string and convert all letters to lower case
    text = re.sub(r'[^a-z\s]', '', text) #deleting every char that is not in alphabet or space
    return text


print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH) #reading the csv file
val_df = pd.read_csv(VAL_PATH) #reading the csv file

# Register tqdm for pandas
tqdm.pandas(desc="Cleaning Text") #enabling to use text in progress bar
train_df['clean_text'] = train_df['text'].progress_apply(preprocess_text) #apply preprocess function on train set
val_df['clean_text'] = val_df['text'].progress_apply(preprocess_text)  #apply preprocess function on validation set

# ==========================================
# 3. Tokenization & Padding
# ==========================================
print("Tokenizing...")
#create a token object (remember the top max words which are the most frequent words)
#if the model sees a word that is not inside the vocabulary it repalce it with OOV (out of vocabulary) so it won't crush
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['clean_text']) #mapping words to numbers based on frequency

# Convert text to sequences
# convert each text sentence into a sequence of numbers based on the mapping we provided earlier
X_train_seq = tokenizer.texts_to_sequences(train_df['clean_text'])
X_val_seq = tokenizer.texts_to_sequences(val_df['clean_text'])

# Pad sequences (crucial for batch training)
#every row will have MAX_LEN tokens
#if there is less it will pad with zeros at the end of the list
#if there is more than it will truncate the last elements to make sure the size is MAX_LEN.
X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_val = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding='post', truncating='post')

# Encode Labels (0-5)
le = LabelEncoder() #makes an instance of the LabelEncoder class
y_train = le.fit_transform(train_df['label']) #Learns all unique categories and then transform each category into a unique number
y_val = le.transform(val_df['label'])


# But ONLY top 10,000 get unique indices (1 through 10,000)
# Words ranked 10,001-15,214 → all become <OOV> token
print(f"Vocabulary Size: {len(tokenizer.word_index) + 1}") # = 15214
print(f"Training Shape: {X_train.shape}, Validation Shape: {X_val.shape}")

#Embeding
print("Loading GloVe Twitter embeddings...")
embeddings_index = {}
with open(GLOVE_PATH, encoding="utf8") as f:
    for line in tqdm(f, desc="Loading GloVe vectors"):
        values = line.rstrip().split(" ") #removes the newline at the end.  splits the line into a list on spaces.
        word = values[0] #value[0] is the word
        coefs = np.asarray(values[1:], dtype="float32") #values[1:] are all the numbers after the word (the 100 embedding components).
        embeddings_index[word] = coefs #Stores the mapping in the dictionary.


vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
embedding_matrix = np.zeros((vocab_size, EMBED_DIM), dtype="float32")

for word, idx in tqdm(tokenizer.word_index.items(), total=len(tokenizer.word_index), desc="Building embedding matrix"):
    if idx >= vocab_size:
        continue
    vec = embeddings_index.get(word) #tries to find that word’s pretrained vector.
    #If found, we put it into the correct row of embedding_matrix:
    #row idx gets the GloVe vector.
    #If not found, that row stays all zeros (unknown in GloVe).
    if vec is not None:
        embedding_matrix[idx] = vec

# ==========================================
# 4. Model Building Function
# ==========================================
def build_model(model_type='lstm', units=64, dropout=0.3):
    """
    Builds a Keras model with proper Masking and Regularization.
    mask_zero=True tells the LSTM to ignore padding (0s), which improves accuracy significantly.
    """
    model = Sequential() #create the model so when can later stuck layers on it

    model.add(Embedding(
        input_dim=embedding_matrix.shape[0],  # vocab_size
        output_dim=embedding_matrix.shape[1],  # EMBED_DIM - Each word index is mapped to an embedding vector of this dimension
        weights=[embedding_matrix],
        input_length=MAX_LEN, #This sets the expected length of each input sequence
        mask_zero=True, #This tells the layer to ignore/pad zeros (used to fill out short sequences).
        trainable=False  # or True if you want to fine-tune
    ))

    # Spatial Dropout drops entire 1D feature maps (better for NLP)
    model.add(SpatialDropout1D(dropout))
    #forcing the model not to over‑rely on any single embedding dimension and helping reduce overfitting in NLP

    # Recurrent Layer (Bidirectional allows learning from future context)
    
    model.add(Bidirectional(LSTM(units, return_sequences=False)))
    #return_sequences=False means the layer outputs just the final vector (not the output from each time step).

    # Output Layer
    model.add(Dense(6, activation='softmax')) #there are 6 classes so we use 6 output units

    model.compile(optimizer='adam', #Adam is the algorithm that updates the model's weights to reduce error.
                  loss='sparse_categorical_crossentropy', #This specific loss is used for multi-class classification when your labels are integers
                  metrics=['accuracy']) #make sure that our metric to evaluate the model results is accuracy
    return model


# ==========================================
# 5. Training Loop with TQDM & Callbacks
# ==========================================

results = []

print("\n" + "=" * 60)
print("LSTM HYPERPARAMETER GRID SEARCH")
print("=" * 60)

units_list = [16, 32, 64, 128, 256]
dropout_list = np.arange(0.05, 0.55, 0.05)  # 0.05, 0.10, 0.15, ..., 0.50

for units in tqdm(units_list, desc="Units loop", leave=True):
    for dropout in tqdm(dropout_list, desc=f"Dropout for units={units}", leave=False):

        model = build_model(model_type='lstm', units=units, dropout=dropout)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=0)
        ]

        # manual epoch loop so tqdm shows training progress
        history = {'val_accuracy': []}
        pbar = tqdm(range(EPOCHS), desc=f"Train (units={units}, drop={dropout:.2f})", leave=False)

        for epoch in pbar:
            hist = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=1,
                batch_size=BATCH_SIZE,
                callbacks=callbacks,
                verbose=0
            )

            val_acc = hist.history['val_accuracy'][0]
            history['val_accuracy'].append(val_acc)
            pbar.set_postfix({'val_acc': f"{val_acc:.4f}"})

        best_acc = max(history['val_accuracy'])
        results.append({'units': units, 'dropout': dropout, 'val_acc': best_acc})
        print(f"  → Best Val Acc: {best_acc:.4f}")

# Print final summary table
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
results.sort(key=lambda x: x['val_acc'], reverse=True)  # Sort by best accuracy

for i, res in enumerate(results, 1):
     print(f"{i:2d}. LSTM | Units: {res['units']:3d} | Dropout: {res['dropout']:.2f} | Val Acc: {res['val_acc']:.4f}")
