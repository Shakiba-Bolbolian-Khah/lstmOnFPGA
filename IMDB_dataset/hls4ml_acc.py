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
# import plotting
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import hls4ml



# def plot(a, b=None):
#     plt.plot(a, label='a')
#     if b is not None:
#         plt.plot(b, label='b')
#     plt.legend()
#     plt.show()

# fix random seed for reproducibility
np.random.seed(9)
OUTPUT_DIR = 'rnn_hls4ml/hls4ml_prj/'
glove_dir =  '/home/skhah/hls4ml/'
# '/home/shakiba/Documents/Lab_Works/Scripts/IMDB_dataset/'
# '/home/skhah/hls4ml/'


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



x_test, _, y_test, _ = train_test_split(
    x_test, y_test, test_size=25000 - 1000, random_state=42, stratify=y_test
)

# Pad sequences
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)
print(f"Padded training data shape: {x_train.shape}")
print(f"Padded test data shape: {x_test.shape}")

# Load GloVe embeddings (assuming glove.6B.100d.txt is downloaded)
embedding_index = {}
with open(glove_dir + 'glove.6B.100d.txt', encoding='utf-8') as f:
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

# Generate embedded training and test data using the embedding model
x_train_embedded = embedding_model.predict(x_train)
x_test_embedded = embedding_model.predict(x_test)
print(f"Embedded training data shape: {x_train_embedded.shape}")
print(f"Embedded test data shape: {x_test_embedded.shape}")


# Fix input shape for LSTM
# Ensure data is 3D (batch_size, timesteps, features)
x_train_embedded = np.reshape(x_train_embedded, (x_train_embedded.shape[0], max_length, embedding_dim))
x_test_embedded = np.reshape(x_test_embedded, (x_test_embedded.shape[0], max_length, embedding_dim))

# Model 2: LSTM Model
model = Sequential()
model.add(LSTM(128, input_shape=(max_length, embedding_dim), name='lstm1'))
model.add(Dropout(0.2, name='dropout'))
model.add(Dense(1, activation='sigmoid', name='output'))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

# Define Callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=5,  
    mode='max',  
    restore_best_weights=True  
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

model.save('imdb_final_model.h5')  # Entire model
model.save_weights('imdb_final_lstm_weights.weights.h5')  # Weights only
print("Final model and weights saved")


# ============================== HLS4ML ==============================================
config = hls4ml.utils.config_from_keras_model(model, granularity='model', default_precision='fixed<16,6>')
# print(config)
# print("Config-----------------------------------")
# plotting.print_dict(config)
# print("-----------------------------------------")

hls_model = hls4ml.converters.convert_from_keras_model(
         model,
         hls_config=config,
         output_dir='rnn_hls4ml/hls4ml_prj',
         backend = 'Quartus',
         part='10AX115N2F40E2LG',
         clock_period = 5)
         #part='xcku115-flvb2104-2-')
         #part='xcu250-figd2104-2L-e')


hls_model.compile()

# ============================== Storing Test Data ====================================


# Save as .npy (if needed later)
np.save(OUTPUT_DIR+"tb_data/inputs.npy", x_test)
np.save(OUTPUT_DIR+"tb_data/outputs.npy", y_test)

# Convert to .dat format (required for io_parallel mode)
with open(OUTPUT_DIR+"tb_data/tb_input_features.dat", "w") as f:
    for row in x_test.reshape(x_test.shape[0], -1):  # Flatten if necessary
        f.write(" ".join(map(str, row)) + "\n")

with open(OUTPUT_DIR+"tb_data/tb_output_predictions.dat", "w") as f:
    for row in y_test.reshape(y_test.shape[0], -1):  # Flatten if necessary
        f.write(" ".join(map(str, row)) + "\n")

# ============================== Predicting and Building the Model======================

y_keras = model.predict(np.ascontiguousarray(x_test_embedded))
y_hls = hls_model.predict(np.ascontiguousarray(x_test_embedded))

y_keras_labels = (y_keras >= 0.5).astype(int)
y_hls_labels = (y_hls >= 0.5).astype(int)

print("Keras  Accuracy: {}".format(accuracy_score(y_test, y_keras_labels)))
print("hls4ml  Accuracy: {}".format(accuracy_score(y_test, y_hls_labels)))

# hls_model.build(synth=True, fpgasynth=True, log_level=1, cont_if_large_area=True)


