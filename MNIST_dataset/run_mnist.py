import sys
import os

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add parent directory to sys.path
sys.path.append(parent_dir)

from fixedPointLStmWeight import *
from sparseLSTM import SparseSHIR_LSTM
from sparseLSTM import *
from fixedPointLStmWeight import SHIR_LSTM
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import Model

#from model import mnist_lstm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# fix random seed for reproducibility
np.random.seed(9)
OUTPUT_DIR = './mnist/'
# input dimension 
n_input = 28
# timesteps
n_step = 28
# output dimension
n_classes = 10

# load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Select a subset of 100 samples from the test set

#x_test, _, y_test, _ = train_test_split(
#     x_test, y_test, test_size=0.99, random_state=42, stratify=y_test
#)

print(f"x_test_subset shape: {x_test.shape}")
print(f"y_test_subset shape: {y_test.shape}")

# prepare the dataset for training and testing
# reshape input to be  [samples, time steps, features]
x_train = x_train.reshape(-1, n_step, n_input)
x_test = x_test.reshape(-1, n_step, n_input)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
 
y_train = to_categorical(y_train, n_classes)
y_test  = to_categorical(y_test,  n_classes)

# lstm = SHIR_LSTM(28,32,28,10)
# y = lstm.run_LSTM(OUTPUT_DIR, x_test, is_input_file = True, test_for_accuracy = True)

y_keras = load_matrix("y_keras", OUTPUT_DIR)
# np.savetxt(OUTPUT_DIR + 'y_label.csv', np.argmax(y, axis=1), fmt='%i', delimiter=',')
# np.savetxt(OUTPUT_DIR + 'y_keras_label.csv', np.argmax(y_keras, axis=1), fmt='%i', delimiter=',')


pruning_dic = {'f':[1,0.75], 'o':[1,0.75], 'i':[1,0.75], 'c':[1,0.75]}
block_number = {'w': 4, 'u': 4}

print("Sparsity Degree: ", pruning_dic)

lstm = SparseSHIR_LSTM(28, 32, 28, 1, sparsity=True, what_to_prune= pruning_dic, block_numbers=block_number)  # 20% row-wise sparsity

lstm.save_pruned_weights(OUTPUT_DIR)
y = lstm.run_LSTM(OUTPUT_DIR, x_test, is_input_file = True, test_for_accuracy = True, dense_activation = "sigmoid")


print("Keras  Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))
print("SHIR  Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y, axis=1))))

