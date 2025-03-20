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




def plot(a, b=None):
    plt.plot(a, label='a')
    if b is not None:
        plt.plot(b, label='b')
    plt.legend()
    plt.show()

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


epochs = 10

# Load the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
print(f"Training data shape of x: {x_train.shape}")
print(f"Training data shape of y: {y_train.shape}")
print(f"Test data shape of x: {x_test.shape}")


# x_test, _, y_test, _ = train_test_split(
#     x_test, y_test, test_size=0.99, random_state=42, stratify=y_test
# )

# Pad sequences
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)
print(f"Padded training data shape: {x_train.shape}")
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


# Model 2: LSTM Classifier Model
input_shape = (max_length, embedding_dim)  # Input shape from embedding output
inputs = Input(shape=input_shape, name='input_layer')
x = LSTM(128, name='lstm1')(inputs)
x = Dropout(0.2, name='dropout')(x)
outputs = Dense(1, activation='sigmoid', name='output')(x)
model = Model(inputs=inputs, outputs=outputs, name='lstm_classifier')
model.summary()

# Compile the LSTM model
optimizer = Adam(learning_rate=0.001)  # Explicit learning rate
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Generate embedded training and test data using the embedding model
x_train_embedded = embedding_model.predict(x_train)
x_test_embedded = embedding_model.predict(x_test)
print(f"Embedded training data shape: {x_train_embedded.shape}")
print(f"Embedded test data shape: {x_test_embedded.shape}")

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=5,              # Stop after 3 epochs without improvement
    mode='max',              # Maximize accuracy
    restore_best_weights=True  # Restore weights from best epoch
)

checkpoint = ModelCheckpoint(
    'best_imdb_model.h5',    # Save best model to HDF5 file
    monitor='val_accuracy',  # Save based on val_accuracy
    save_best_only=True,     # Only save if improved
    mode='max'
)

# Train the LSTM model on embedded data
batch_size = 32
model.fit(    
    x_train_embedded, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,    # 10% of training data for validation
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Evaluate the LSTM model on the test set
test_loss, test_accuracy = model.evaluate(x_test_embedded, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

# model.save('imdb_final_model.h5')  # Entire model
# model.save_weights('imdb_final_lstm_weights.weights.h5')  # Weights only
# print("Final model and weights saved")


# ============================= Storing Weight Matrices for SHIR implementation =====

lstm_layer = model.get_layer('lstm1')
kernel, recurrent_kernel, bias = lstm_layer.get_weights()

print('kernel:', kernel.shape)

units = lstm_layer.units  # In your model, this is 100

weights = {}

biases = {}

# Split the kernel into four parts (each of shape: (input_dim, units))
weights['wi'], weights['wf'], weights['wc'], weights['wo'] = np.split(kernel, 4, axis=1)

# Split the recurrent kernel into four parts (each of shape: (units, units))
weights['ui'], weights['uf'], weights['uc'], weights['uo'] = np.split(recurrent_kernel, 4, axis=1)

# Split the bias vector into four parts (each of shape: (units,))
biases['bi'], biases['bf'], biases['bc'], biases['bo'] = np.split(bias, 4)

# Now you have the weights for each gate:
print("Input Gate Kernel shape:", weights['wi'].shape)
print("Forget Gate Recurrent Kernel shape:", weights['ui'].shape)
print("Cell Gate Bias shape:", biases['bi'].shape)

for weight in weights:
    quantize_matrix(weights[weight], weight, OUTPUT_DIR)

for b in biases: 
    quantize_matrix(biases[b].reshape(units,1), b, OUTPUT_DIR)
    
dense_layer = model.get_layer('output')
dense_kernel, dense_bias = dense_layer.get_weights()

quantize_matrix(dense_kernel, "wd", OUTPUT_DIR)
quantize_matrix(dense_bias, "bd", OUTPUT_DIR)

