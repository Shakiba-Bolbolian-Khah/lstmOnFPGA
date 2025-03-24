import sys
import os

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add parent directory to sys.path
sys.path.append(parent_dir)

from fixedPointLStmWeight import *
from fixedPointLStmWeight import SHIR_LSTM

# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Parameters
timesteps = 128    # Sequence length
n_features = 1     # Univariate (1 feature per timestep) n_input
n_classes = 4      # 4 patterns
batch_size = 32
epochs = 50

OUTPUT_DIR = 'twopattern/'

# Load Two_Patterns dataset (assumes downloaded from UCR and placed in 'UCR_Two_Patterns/')
def load_two_patterns_data():
    base_path = 'Data_TwoPatterns/'
    header_lines = 15  # 7 comment lines + 8 metadata lines including @data

    # Helper function to parse rows with [t1,t2,...,t128:label] format
    def parse_file(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()[header_lines:]  # Skip header
        x_data = []
        y_data = []
        for line in lines:
            # Split on ':' to separate timesteps and label
            parts = line.strip().split(':')
            if len(parts) != 2:
                continue  # Skip malformed lines
            timesteps_str, label = parts
            # Split timesteps on ',' and convert to float
            timesteps_values = [float(val) for val in timesteps_str.split(',')]
            if len(timesteps_values) != timesteps:
                continue  # Skip rows with wrong timestep count
            x_data.append(timesteps_values)
            y_data.append(float(label) - 1)  # Labels 1-4 to 0-3
        return np.array(x_data).reshape(-1, timesteps, n_features), np.array(y_data)

    # Load training and test data
    x_train, y_train = parse_file(base_path + 'TwoPatterns_TRAIN.ts')
    x_test, y_test = parse_file(base_path + 'TwoPatterns_TEST.ts')

    return x_train, y_train, x_test, y_test

# Load data
x_train, y_train, x_test, y_test = load_two_patterns_data()
print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")

# Build the model using Sequential and model.add()
model = Sequential()
model.add(LSTM(32, input_shape=(timesteps, n_features), name='lstm1'))  # Single LSTM, no activation
model.add(Dropout(0.2, name='dropout_1'))
model.add(Dense(16, name='dense_1'))  # No activation (linear)
model.add(Dropout(0.2, name='dropout_2'))
model.add(Dense(n_classes, activation='softmax', name='dense_2'))  # Output layer with softmax
model.summary()

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    mode='max',
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_two_patterns_lstm_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Train the model with validation split
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

# Save final model and weights
# model.save('final_two_patterns_lstm_model.keras')
# model.save_weights('final_two_patterns_lstm_weights.weights.h5')
print("Final model saved to 'final_two_patterns_lstm_model.keras'")
print("Weights saved to 'final_two_patterns_lstm_weights.weights.h5'")


# ============================= Storing Weight Matrices for SHIR implementation =====

lstm_layer1 = model.get_layer('lstm1')
kernel, recurrent_kernel, bias = lstm_layer1.get_weights()

print('kernel:', kernel.shape)

units1 = lstm_layer1.units  # In your model, this is 100

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
    quantize_matrix(biases[b].reshape(-1,1), b, OUTPUT_DIR)
    
dense_layer1 = model.get_layer('dense_1')
dense_kernel1, dense_bias1 = dense_layer1.get_weights()

quantize_matrix(dense_kernel1, "wd1", OUTPUT_DIR)
quantize_matrix(dense_bias1, "bd1", OUTPUT_DIR)

dense_layer2 = model.get_layer('dense_2')
dense_kernel2, dense_bias2 = dense_layer2.get_weights()

quantize_matrix(dense_kernel2, "wd2", OUTPUT_DIR)
quantize_matrix(dense_bias2, "bd2", OUTPUT_DIR)


quantize_input(x_test, 1, 'x_test', OUTPUT_DIR)
quantize_matrix(y_test, 'y_test', OUTPUT_DIR, need_transpose = False)
