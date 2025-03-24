import sys
import os

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add parent directory to sys.path
sys.path.append(parent_dir)


from fixedPointLStmWeight import *
from fixedPointLStmWeight import SHIR_LSTM
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
tf.random.set_seed(0)
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers, backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

TRAIN_RANGE = (datetime(1949, 1, 1), datetime(1956, 12, 1))
VALID_RANGE = (datetime(1957, 1, 1), datetime(1960, 10, 1))
TEST_RANGE = (datetime(1960, 11, 1), datetime(1960, 12, 1))
TIMESTEPS = 12  # Input 12 months to predict next month

timesteps = 12    # Sequence length
n_features = 1     # Univariate (1 feature per timestep) n_input
n_classes = 1      # 1 value


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url)
data.index = pd.to_datetime(data.Month)  # Set datetime index
data.drop(['Month'], axis=1, inplace=True)


OUTPUT_DIR = 'flight/'


def run_two_pattern(self, dir, test_for_accuracy = False):
        self.load_weights(dir)
        self.load_biases(dir)
        self.generate_initial_state(dir)
        self.load_dense_layer(dir, 2)
        self.load_input(dir)

        print(self.x.shape)

        y = []
        for item in self.x:
            lstm = self.run_inference(item, False, True)
            dense1 = self.run_dense(lstm, '1')
            dense2 = self.run_dense(dense1, '2')

            y+= [dense2]
        
        print(type(y[0]))
        y = np.array(y)
        np.savetxt(dir + 'y.csv', y.astype(int_type(BIT_WIDTH)), fmt='%i', delimiter=',')
        return(y)


lstm1 = SHIR_LSTM(n_features,64,timesteps, n_classes)
lstm2 = SHIR_LSTM(64, 32, timesteps, n_classes)

dir1 = 'flight1/'
dir2 = 'flight2/'

def run_flight():
    lstm1.load_weights(dir1)
    lstm1.load_biases(dir1)
    lstm1.generate_initial_state(dir1)
    lstm1.load_input(dir2)

    lstm2.load_weights(dir2)
    lstm2.load_biases(dir2)
    lstm2.generate_initial_state(dir2)
    lstm2.load_dense_layer(dir2, 1)

    y = []
    for item in lstm1.x:
        lstm1_out = lstm1.run_inference(item, True, True)
        lstm2_out = lstm2.run_inference(lstm1_out, False, True)
        dense1 = lstm2.run_dense(lstm2_out)

        y+= [dense1]
    
    print(type(y[0]))
    y = np.array(y)
    np.savetxt(dir2 + 'y_out.csv', y.astype(int_type(BIT_WIDTH)), fmt='%i', delimiter=',')
    return(y)

run_flight()

