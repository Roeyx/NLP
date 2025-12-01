import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. Setup & Configuration
# ==========================================
TRAIN_PATH = r"C:\Users\roeyn\Coding_Enviroment\NLP\Ex1\.venv\train.csv"
VAL_PATH = r"C:\Users\roeyn\Coding_Enviroment\NLP\Ex1\.venv\validation.csv"

# Hyperparameters
MAX_WORDS = 10000  # Max vocabulary size
MAX_LEN = 100  # Max sequence length
EMBED_DIM = 100  # Embedding dimension
EPOCHS = 20
BATCH_SIZE = 32


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

print(f"Vocabulary Size: {len(tokenizer.word_index) + 1}")
print(f"Training Shape: {X_train.shape}, Validation Shape: {X_val.shape}")


# ==========================================
# 4. Model Building Function
# ==========================================
def build_model(model_type='lstm', units=64, dropout=0.3):
    """
    Builds a Keras model with proper Masking and Regularization.
    mask_zero=True tells the LSTM to ignore padding (0s), which improves accuracy significantly.
    """
    model = Sequential() #create the model so when can later stuck layers on it

    # Embedding Layer: mask_zero=True is key for variable length sequences!
    model.add(Embedding(input_dim=min(MAX_WORDS, len(tokenizer.word_index) + 1),
                        output_dim=EMBED_DIM,
                        input_length=MAX_LEN,
                        mask_zero=True))

    # Spatial Dropout drops entire 1D feature maps (better for NLP)
    model.add(SpatialDropout1D(dropout))

    # Recurrent Layer (Bidirectional allows learning from future context)
    if model_type == 'lstm':
        model.add(Bidirectional(LSTM(units, return_sequences=False)))
    else:
        model.add(Bidirectional(GRU(units, return_sequences=False)))

    # Output Layer
    model.add(Dense(6, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ==========================================
# 5. Training Loop with TQDM & Callbacks
# ==========================================
configs = [
    {'type': 'lstm', 'units': 64, 'dropout': 0.2},
    {'type': 'lstm', 'units': 128, 'dropout': 0.4},
]

results = []

print("\nStarting Training Loop...")
for config in configs:
    print(f"\nTraining {config['type'].upper()} (Units: {config['units']}, Dropout: {config['dropout']})")

    model = build_model(model_type=config['type'], units=config['units'], dropout=config['dropout'])

    # Callbacks to prevent overfitting and stuck learning
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    ]

    # Keras handles the progress bar automatically, but we can wrap the fit call if needed.
    # Standard Keras output is usually clear enough, but here is the standard execution.
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    best_acc = max(history.history['val_accuracy'])
    results.append({'config': config, 'val_acc': best_acc})

# ==========================================
# 6. Final Results
# ==========================================
print("\n" + "=" * 50)
print("FINAL RESULTS SUMMARY")
print("=" * 50)
for res in results:
    c = res['config']
    print(
        f"Model: {c['type'].upper()} | Units: {c['units']:3d} | Dropout: {c['dropout']:.1f} | Val Acc: {res['val_acc']:.4f}")

