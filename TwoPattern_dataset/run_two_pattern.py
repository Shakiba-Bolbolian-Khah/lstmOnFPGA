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



x_train, y_train, x_test, y_test = load_two_patterns_data()
print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")



# Select a subset of 100 samples from the test set

# x_test, _, y_test, _ = train_test_split(
#     x_test, y_test, test_size=4000-4, random_state=42, stratify=y_test
# )

print(f"x_test_subset shape: {x_test.shape}")
print(f"y_test_subset shape: {y_test.shape}")



quantize_input(x_test, n_features, 'x', OUTPUT_DIR)


lstm = SHIR_LSTM(n_features,32,timesteps, n_classes)
y = lstm.run_two_dense(OUTPUT_DIR)

y_pred_labels = np.argmax(y, axis=1)

print("SHIR  Accuracy: {}".format(accuracy_score(y_test, y_pred_labels)))
