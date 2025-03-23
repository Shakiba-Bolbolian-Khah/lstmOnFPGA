import sys
import os

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add parent directory to sys.path
sys.path.append(parent_dir)

from fixedPointLStmWeight import *
from fixedPointLStmWeight import SHIR_LSTM
from tensorflow.keras.datasets import mnist, imdb
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Activation, Embedding, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Lambda
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
import tensorflow as tf

#from model import mnist_lstm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# fix random seed for reproducibility
np.random.seed(9)
OUTPUT_DIR = 'imdb/'

# Parameters
num_words = 10000  # Top 10,000 words
max_length = 200   # Pad/truncate reviews to 200 words
embedding_dim = 100  # GloVe 100D vectors

# input dimension 
n_input = 100
# timesteps
n_step = 200
# output dimension
n_classes = 2

# load the dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# Select a subset of 100 samples from the test set

x_test, _, y_test, _ = train_test_split(
    x_test, y_test, test_size=25000-1000, random_state=42, stratify=y_test
)

print(f"x_test_subset shape: {x_test.shape}")
print(f"y_test_subset shape: {y_test.shape}")


# Pad sequences
x_test = pad_sequences(x_test, maxlen=max_length)
print(f"Padded test data shape: {x_test.shape}")

# Load GloVe embeddings (assuming glove.6B.100d.txt is downloaded)
embedding_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
print(f"Loaded {len(embedding_index)} GloVe word vectors.")

# Get word index from IMDB dataset and adjust
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# Create embedding matrix
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i < num_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Model 1: Embedding Model
embedding_model = Sequential([
    Embedding(input_dim=num_words, 
              output_dim=embedding_dim, 
              input_length=max_length, 
              weights=[embedding_matrix], 
              trainable=False, 
              name='embedding_layer')
], name='embedding_model')
embedding_model.summary()

x_test_embedded = embedding_model.predict(x_test)
print(f"Embedded test data shape: {x_test_embedded.shape}")


# Fix input shape for LSTM
# Ensure data is 3D (batch_size, timesteps, features)
x_test_embedded = np.reshape(x_test_embedded, (x_test_embedded.shape[0], max_length, embedding_dim))


print(f"Embedded test data shape: {x_test_embedded.shape}")

quantize_input(x_test_embedded, n_input, 'x', OUTPUT_DIR)
quantize_matrix(y_test, 'y_test', OUTPUT_DIR, quantize = False, need_transpose = False)


lstm = SHIR_LSTM(100,128,200,1)
y = lstm.run_LSTM(OUTPUT_DIR, x_test_embedded, is_input_file = False, test_for_accuracy = True, dense_activation = "sigmoid")


y_pred_labels = (y/(2**FRAC_BITS) >= 0.5).astype(int)

print("SHIR  Accuracy: {}".format(accuracy_score(y_test, y_pred_labels)))
