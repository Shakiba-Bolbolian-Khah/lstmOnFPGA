from fixedPointLStmWeight import *
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

def plot(a, b=None):
    plt.plot(a, label='a')
    if b is not None:
        plt.plot(b, label='b')
    plt.legend()
    plt.show()

# fix random seed for reproducibility
np.random.seed(9)
OUTPUT_DIR = 'mnist/'
# input dimension 
n_input = 28
# timesteps
n_step = 28
# output dimension
n_classes = 10

EPOCH = 10

# load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Select a subset of 100 samples from the test set
x_test, _, y_test, _ = train_test_split(
    x_test, y_test, test_size=0.89, random_state=42, stratify=y_test
)

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

# np.save(OUTPUT_DIR+ 'tb_data/tb_input_features.dat', x_test)
# np.save(OUTPUT_DIR+ 'tb_data/tb_output_predictions.dat', y_test)

# load the model
model = Sequential()
model.add(LSTM(32, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2]), name='lstm1'))
#model.add(Dense(16, activation='relu', name='fc1'))
model.add(Dense(10, activation='softmax', name='output'))
# model.add(Dense(10, activation=None, name='output'))

adam = Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# train the model 
model.fit(x_train, y_train,
          batch_size=128,
          epochs=EPOCH,
          verbose=1,
          #callbacks=[lstm_es, lstm_mc],
          validation_split=0.2)

# evaluate the model 
scores = model.evaluate(x_test, y_test, verbose=0)
print('LSTM test score:', np.round(scores[0]))
print('LSTM test accuracy:', np.round(scores[1]))

# ============================= Storing Weight Matrices for SHIR implementation =====

lstm_layer = model.get_layer('lstm1')
kernel, recurrent_kernel, bias = lstm_layer.get_weights()

print('kernel:', kernel.shape)

units = lstm_layer.units  # In your model, this is 32

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
quantize_matrix(dense_bias.reshape(n_classes,1), "bd", OUTPUT_DIR)


# ============================== Deleting Softmax from Dense Layer ===================

inference_input = model.input
lstm_output = model.get_layer('lstm1').output

dense_linear = Dense(10, activation=None, name='output_linear')
dense_linear.build(lstm_output.shape)
dense_linear.set_weights(model.get_layer('output').get_weights())

dense_out = dense_linear(lstm_output)

# argmax_output = Lambda(lambda x: K.argmax(x, axis=-1), name='argmax')(dense_out)

# Create the inference model
inference_model = Model(inputs=inference_input, outputs=dense_out)

adam = Adam(learning_rate=0.001)
inference_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
inference_model.summary()

# Now you can use inference_model for prediction
# Re-evaluate the model 
inference_predicted_classes = inference_model.predict(x_test)
predicted_classes = model.predict(x_test)

inference_acc = np.argmax(inference_predicted_classes, axis=1)
softmax_acc = np.argmax(predicted_classes, axis=1)
print("Matchinging (%) output of model with/without softmax", (sum(inference_acc==softmax_acc)/inference_acc.shape[0])*100)

# ============================== Storing Test Data ====================================


# Save as .npy (if needed later)

quantize_input(x_test, n_input, 'x', OUTPUT_DIR)
quantize_matrix(y_test, 'y_test', OUTPUT_DIR, need_transpose = False)

y_keras = inference_model.predict(np.ascontiguousarray(x_test))

quantize_matrix(y_test, 'y_keras', OUTPUT_DIR, need_transpose = False)