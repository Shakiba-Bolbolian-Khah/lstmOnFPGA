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
VALID_RANGE = (datetime(1957, 1, 1), datetime(1960, 5, 1))
TEST_RANGE = (datetime(1960, 6, 1), datetime(1960, 12, 1))
TIMESTEPS = 12  # Input 12 months to predict next month



url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url)
data.index = pd.to_datetime(data.Month)  # Set datetime index
data.drop(['Month'], axis=1, inplace=True)


OUTPUT_DIR = 'flight/'

# data.plot(figsize=(14, 8), title='Monthly airline passengers')

scaler = MinMaxScaler(feature_range=(0, 1))
data['NormalizedPassengers'] = scaler.fit_transform(data['Passengers'].values.reshape(-1, 1)).flatten()
# data[['NormalizedPassengers']].plot(figsize=(14, 8), title='Monthly normalized airline passengers')

def create_dataset(data, timesteps=TIMESTEPS):
    """Create input and output pairs for training lstm.
    Params:
        data (pandas.DataFrame): Normalized dataset
        timesteps (int, default: TIMESTEPS): Input time length
    Returns:
        X (numpy.array): Input for lstm
        y (numpy.array): Output for lstm
        y_date (list): Datetime of output
        start_values (list): Start valeus of each input
    """
    X, y, y_date, start_values = [], [], [], []

    for i in range(len(data) - timesteps):
        Xt = data.iloc[i:i+timesteps].values
        yt = data.iloc[i+timesteps]
        yt_date = data.index[i+timesteps].to_pydatetime()

        # Subtract a start value from each values in the timestep.
        start_value = Xt[0]
        Xt = Xt - start_value
        yt = yt - start_value

        X.append(Xt)
        y.append(yt)
        y_date.append(yt_date)
        start_values.append(start_value)

    return np.array(X), np.array(y), y_date, start_values

def split_train_valid_test(X, y, y_date, train_range=TRAIN_RANGE, valid_range=VALID_RANGE, test_range=TEST_RANGE):
    """Split X and y into train, valid, and test periods.
    Params:
        X (numpy.array): Input for lstm
        y (numpy.array): Output for lstm
        y_date (list): Datetime of output
        train_range (tuple): Train period
        valid_range (tuple): Validation period
        test_range (tuple): Test period
    Returns:
        X_train (pandas.DataFrame)
        X_valid (pandas.DataFrame)
        X_test (pandas.DataFrame)
        y_train (pandas.DataFrame)
        y_valid (pandas.DataFrame)
        y_test (pandas.DataFrame)
        y_date_train (list)
        y_date_valid (list)
        y_date_test (list)
    """
    train_end_idx = y_date.index(train_range[1])
    valid_end_idx = y_date.index(valid_range[1])

    X_train = X[:train_end_idx+1, :]
    X_valid = X[train_end_idx+1:valid_end_idx+1, :]
    X_test = X[valid_end_idx+1:, :]

    y_train = y[:train_end_idx+1]
    y_valid = y[train_end_idx+1:valid_end_idx+1]
    y_test = y[valid_end_idx+1:]

    y_date_train = y_date[:train_end_idx+1]
    y_date_valid = y_date[train_end_idx+1:valid_end_idx+1]
    y_date_test = y_date[valid_end_idx+1:]

    return X_train, X_valid, X_test, y_train, y_valid, y_test, y_date_train, y_date_valid, y_date_test

# Create input and output pairs for training lstm.
X, y, y_date, start_values = create_dataset(data[['NormalizedPassengers']])

# Split X and y into train, valid, and test periods.
X_train, X_valid, X_test, y_train, y_valid, y_test, y_date_train, y_date_valid, y_date_test = split_train_valid_test(X, y, y_date)


print(X_train.shape, X_valid.shape, X_test.shape)
print(y_train.shape, y_valid.shape, y_test.shape)

def create_model(timesteps=TIMESTEPS):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(timesteps, 1), name='lstm1'))  # Input timesteps months with scalar value.
    model.add(LSTM(32, name='lstm2'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=0.01), metrics=['mean_absolute_error'])
    return model

# Create model
model = create_model()
model.summary()

# Callbacks of training.
es = EarlyStopping(monitor='val_mean_absolute_error', min_delta=0, patience=5, verbose=1, mode='auto')
fn = 'best_airlines_lstm_dense_model.h5'
mc = ModelCheckpoint(filepath=fn, save_best_only=True)
callbacks = [es, mc]

# Start training model.
fit = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    verbose=2,
    validation_data=(X_valid, y_valid),
    callbacks=callbacks)



# Load best model
model = load_model(fn)

def evaluate_model(data, scaler, X_train, X_valid, X_test, y_train, y_valid, y_test, y_date_train, y_date_valid, y_date_test, start_values, model):
    """Evaluate trained model by rmse (root mean squared error) and mae (mean absolute error)'"""

    # Predict next month passengers
    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)

    # Add start_values that were subtracted when preprocessing.
    pred_train  = pred_train + start_values[:len(X_train)]
    pred_valid  = pred_valid + start_values[len(X_train):len(X_train)+len(X_valid)]
    pred_test  = pred_test + start_values[len(X_train)+len(X_valid):]

    # Inverse transform normalization
    pred_train = scaler.inverse_transform(pred_train).flatten()
    pred_valid = scaler.inverse_transform(pred_valid).flatten()
    pred_test = scaler.inverse_transform(pred_test).flatten()

    pred_df = data.copy()
    pred_df.loc[y_date_train[0]:y_date_train[-1], 'PredictionTrain'] = pred_train
    pred_df.loc[y_date_valid[0]:y_date_valid[-1], 'PredictionValid'] = pred_valid
    pred_df.loc[y_date_test[0]:y_date_test[-1], 'PredictionTest'] = pred_test
    pred_df[['Passengers', 'PredictionTrain', 'PredictionValid', 'PredictionTest']].plot(figsize=(12, 6), title='Predicted monthly airline passengers')
    # fig.show()

    # Add start_values that were subtracted when preprocessing.
    y_train  = y_train + start_values[:len(X_train)]
    y_valid  = y_valid + start_values[len(X_train):len(X_train)+len(X_valid)]
    y_test  = y_test + start_values[len(X_train)+len(X_valid):]

    # Inverse transform normalization
    y_train = scaler.inverse_transform(y_train).flatten()
    y_valid = scaler.inverse_transform(y_valid).flatten()
    y_test = scaler.inverse_transform(y_test).flatten()

    # Evaluate prediction scores of model.
    for y, pred, mode in zip([y_train, y_valid, y_test], [pred_train, pred_valid, pred_test], ['train', 'valid', 'test']):
        rmse = np.sqrt(mean_squared_error(y, pred))
        mae = mean_absolute_error(y, pred)
        print(f'{mode} rmse: {rmse:.06f}, mae: {mae:.06f}')

evaluate_model(data, scaler, X_train, X_valid, X_test, y_train, y_valid, y_test, y_date_train, y_date_valid, y_date_test, start_values, model)


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
    quantize_matrix(weights[weight], weight, 'flight1/')

for b in biases: 
    quantize_matrix(biases[b].reshape(-1,1), b, 'flight1/')


# =========================================================


lstm_layer2 = model.get_layer('lstm2')
kernel, recurrent_kernel, bias = lstm_layer2.get_weights()

print('kernel:', kernel.shape)

units = lstm_layer2.units  # In your model, this is 100


# Split the kernel into four parts (each of shape: (input_dim, units))
weights['wi'], weights['wf'], weights['wc'], weights['wo'] = np.split(kernel, 4, axis=1)

# Split the recurrent kernel into four parts (each of shape: (units, units))
weights['ui'], weights['uf'], weights['uc'], weights['uo'] = np.split(recurrent_kernel, 4, axis=1)

# Split the bias vector into four parts (each of shape: (units,))
biases['bi'], biases['bf'], biases['bc'], biases['bo'] = np.split(bias, 4)

# Now you have the weights for each gate:
print("Input Gate Kernel shape:", weights['wi'].shape)
print("Forget Gate Recurrent Kernel shape:", weights['uf'].shape)
print("Cell Gate Bias shape:", biases['bc'].shape)


for weight in weights:
    quantize_matrix(weights[weight], weight, 'flight2/')

for b in biases: 
    quantize_matrix(biases[b].reshape(-1,1), b, 'flight2/')
    
dense_layer = model.get_layer('dense')
dense_kernel, dense_bias = dense_layer.get_weights()

quantize_matrix(dense_kernel, "wd",  'flight2/')
quantize_matrix(dense_bias, "bd",  'flight2/')

quantize_input(X_test, 1, 'x',  'flight2/')
quantize_matrix(y_test, 'y',  'flight2/', need_transpose = False)


# ===================================================================================


lstm_layer1 = model.get_layer('lstm1')
kernel, recurrent_kernel, bias = lstm_layer1.get_weights()

print('kernel:', kernel.shape)

units1 = lstm_layer1.units  # In your model, this is 100

weights = {}

biases = {}

# Split the kernel into four parts (each of shape: (input_dim, units))
weights['wi1'], weights['wf1'], weights['wc1'], weights['wo1'] = np.split(kernel, 4, axis=1)

# Split the recurrent kernel into four parts (each of shape: (units, units))
weights['ui1'], weights['uf1'], weights['uc1'], weights['uo1'] = np.split(recurrent_kernel, 4, axis=1)

# Split the bias vector into four parts (each of shape: (units,))
biases['bi1'], biases['bf1'], biases['bc1'], biases['bo1'] = np.split(bias, 4)

# Now you have the weights for each gate:
print("Input Gate Kernel shape:", weights['wi1'].shape)
print("Forget Gate Recurrent Kernel shape:", weights['ui1'].shape)
print("Cell Gate Bias shape:", biases['bi1'].shape)

lstm_layer2 = model.get_layer('lstm2')
kernel, recurrent_kernel, bias = lstm_layer2.get_weights()

print('kernel:', kernel.shape)

units = lstm_layer2.units  # In your model, this is 100


# Split the kernel into four parts (each of shape: (input_dim, units))
weights['wi2'], weights['wf2'], weights['wc2'], weights['wo2'] = np.split(kernel, 4, axis=1)

# Split the recurrent kernel into four parts (each of shape: (units, units))
weights['ui2'], weights['uf2'], weights['uc2'], weights['uo2'] = np.split(recurrent_kernel, 4, axis=1)

# Split the bias vector into four parts (each of shape: (units,))
biases['bi2'], biases['bf2'], biases['bc2'], biases['bo2'] = np.split(bias, 4)

# Now you have the weights for each gate:
print("Input Gate Kernel shape:", weights['wi2'].shape)
print("Forget Gate Recurrent Kernel shape:", weights['uf2'].shape)
print("Cell Gate Bias shape:", biases['bc2'].shape)


for weight in weights:
    quantize_matrix(weights[weight], weight, OUTPUT_DIR)

for b in biases: 
    quantize_matrix(biases[b].reshape(-1,1), b, OUTPUT_DIR)
    
dense_layer = model.get_layer('dense')
dense_kernel, dense_bias = dense_layer.get_weights()

quantize_matrix(dense_kernel, "wd", OUTPUT_DIR)
quantize_matrix(dense_bias, "bd", OUTPUT_DIR)


quantize_input(X_test, 1, 'x', OUTPUT_DIR)
quantize_matrix(y_test, 'y', OUTPUT_DIR, need_transpose = False)

#  ===================================================================================
