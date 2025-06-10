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
import librosa
import os
import pathlib
import zipfile
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Parameters
timesteps = 32    # Time steps for MFCCs
n_features = 13   # Number of MFCC coefficients n_input
n_classes = 8     # 8 commands
batch_size = 32
epochs = 100
sample_rate = 16000  # Audio sample rate
commands = ['yes', 'no', 'up', 'down', 'left', 'right', 'stop', 'go']


OUTPUT_DIR = 'speech/'

# Specify the path to your downloaded mini_speech_commands.zip
zip_path = './mini_speech_commands.zip'  # Replace with your actual path
if not os.path.exists(zip_path):
    raise FileNotFoundError(f"ZIP file {zip_path} does not exist! Please check the path.")
print("Using ZIP file:", zip_path)

# Extract the ZIP file
extract_dir = pathlib.Path('extracted_mini_speech_commands')
if not extract_dir.exists():
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("ZIP file extracted to:", extract_dir)
else:
    print("Extracted directory already exists at:", extract_dir)


# Set data directory
data_dir = extract_dir / 'mini_speech_commands'
if not data_dir.exists():
    raise FileNotFoundError(f"Expected extracted directory {data_dir} not found!")
print("Using data directory:", data_dir)

# Load and preprocess audio files into MFCCs
def load_and_extract_mfcc(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate)
        audio = librosa.util.fix_length(audio, size=sample_rate)  # Ensure 1s
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_features, n_fft=512, hop_length=501)
        return mfcc.T  # Shape: (32, 13)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Prepare dataset
def prepare_dataset(data_dir, commands):
    x_data, y_data = [], []
    for idx, command in enumerate(commands):
        folder = data_dir / command
        files = list(folder.glob('*.wav'))
        if not files:
            print(f"No .wav files found in {folder}")
            continue
        for file in files:
            mfcc = load_and_extract_mfcc(file)
            if mfcc is not None and mfcc.shape == (timesteps, n_features):
                x_data.append(mfcc)
                y_data.append(idx)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    if x_data.size == 0:
        raise ValueError("No valid samples loaded after preprocessing!")
    return x_data, y_data

# Load data
x_data, y_data = prepare_dataset(data_dir, commands)
print(f"Raw data shape: {x_data.shape}, Labels shape: {y_data.shape}")

# Normalize MFCCs
scaler = StandardScaler()
x_data_reshaped = x_data.reshape(-1, n_features)
x_data_normalized = scaler.fit_transform(x_data_reshaped).reshape(x_data.shape)

# Shuffle data
x_data_normalized, y_data = shuffle(x_data_normalized, y_data, random_state=42)

# Split into train and test (80-20)
split_idx = int(len(x_data) * 0.8)
x_train, x_test = x_data_normalized[:split_idx], x_data_normalized[split_idx:]
y_train, y_test = y_data[:split_idx], y_data[split_idx:]
print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")


# Model: LSTM Classifier
model = Sequential()
model.add(LSTM(64, input_shape=(timesteps, n_features), name='lstm1'))
model.add(Dropout(0.5, name='dropout_1'))
model.add(Dense(32, name='dense_1'))  # No activation (linear)
model.add(Dropout(0.5, name='dropout_2'))
model.add(Dense(n_classes, activation='softmax', name='dense_2'))
model.summary()

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', restore_best_weights=True)
checkpoint = ModelCheckpoint('best_speech_commands_lstm_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
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
# model.save('final_speech_commands_lstm_model.keras')
# model.save_weights('final_speech_commands_lstm_weights.weights.h5')
print("Final model saved to 'final_speech_commands_lstm_model.keras'")
print("Weights saved to 'final_speech_commands_lstm_weights.weights.h5'")

# Predictions for confusion matrix and classification report
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)


# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=commands))


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


quantize_input(x_test, n_features, 'x', OUTPUT_DIR)
quantize_matrix(y_test, 'y_label', OUTPUT_DIR, need_transpose = False, quantize= False)


# quantize_input(x_test, n_features, 'x_test', OUTPUT_DIR)
# quantize_matrix(y_test, 'y_test', OUTPUT_DIR, need_transpose = False)


# x_test8, _, y_test8, _ = train_test_split(
#     x_test, y_test, test_size=1600-8, random_state=42, stratify=y_test
# )

# quantize_input(x_test8, n_features, 'x', OUTPUT_DIR)
# quantize_matrix(y_test8, 'y_label', OUTPUT_DIR, need_transpose = False)

