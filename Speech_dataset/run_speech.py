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
from sklearn.model_selection import train_test_split

#from model import mnist_lstm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Parameters
timesteps = 32    # Time steps for MFCCs
n_features = 13   # Number of MFCC coefficients n_input
n_classes = 8     # 8 commands
batch_size = 32
epochs = 100

OUTPUT_DIR = 'speech/'



# quantize_input('x_test', n_features, 'x', OUTPUT_DIR)
y_test = load_matrix("y_label", OUTPUT_DIR)

lstm = SHIR_LSTM(n_features,64,timesteps, n_classes)
y = lstm.run_two_dense(OUTPUT_DIR)

y_pred_labels = np.argmax(y, axis=1)

print("SHIR  Accuracy: {}".format(accuracy_score(y_test, y_pred_labels)))
