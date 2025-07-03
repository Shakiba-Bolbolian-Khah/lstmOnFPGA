import numpy as np
import csv
import os
import sys
import editdistance
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add parent directory to sys.path
sys.path.append(parent_dir)
from fixedPointLStmWeight import *
from sparseLSTM import SparseSHIR_LSTM
from sparseLSTM import *
from fixedPointLStmWeight import SHIR_LSTM
from sparseProjectedLSTM import SparseProjectedLSTM

# Constants
FRAC_BITS = 10
BIT_WIDTH = 16
SCALE = 2 ** FRAC_BITS

OUTPUT_DIR = 'timit/'

# Helper: decode CTC sequence
def ctc_greedy_decode(logits, blank=39):
    best_path = np.argmax(logits, axis=1)
    decoded = []
    prev = -1
    for p in best_path:
        if p != blank and p != prev:
            decoded.append(p)
        prev = p
    return decoded

# Load labels from labels.csv
def load_labels(label_csv):
    all_labels = []
    current_seq = []

    with open(OUTPUT_DIR + label_csv, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if current_seq:
                    all_labels.append(current_seq)
                    current_seq = []
            else:
                current_seq.append(int(line))
    if current_seq:
        all_labels.append(current_seq)
    return all_labels

# Load features from features.csv
def load_features(feature_csv, n_input=153, seq_len=300):
    with open(OUTPUT_DIR + feature_csv, 'r') as f:
        reader = csv.reader(f)
        all_frames = np.array([[int(v) for v in row] for row in reader])

    total_frames = all_frames.shape[0]
    assert total_frames % seq_len == 0, "Total number of frames is not divisible by sequence length."

    sequences = np.split(all_frames, total_frames // seq_len)
    return sequences 

# Compute edit distance based PER
def compute_per(ref, hyp):
    if not ref:
        return 100.0 if hyp else 0.0
    dist = editdistance.eval(ref, hyp)
    return (dist / len(ref)) * 100

# Evaluate model
def evaluate_sparse_projected_lstm(model, feature_csv, label_csv, blank_idx=39, n_input=153, seq_len=300, save_output = False):
    features = load_features(feature_csv, n_input, seq_len)
    # quantize_input(features, n_input, 'x', OUTPUT_DIR)
    labels = load_labels(label_csv)
    y = []
    count = 0

    total_per = 0
    for x, y_true in zip(features, labels):
        
        # Run inference
        logits = model.run_LSTM(dir=OUTPUT_DIR, input=x, is_input_file=False, test_for_accuracy=True)
        y += [logits]
        # Decode
        decoded = ctc_greedy_decode(logits, blank=blank_idx)
        
        print("Decoded vs True:\t", len(decoded), len(y_true))
        # Compute PER
        total_per += compute_per(y_true, decoded)

        count += 1
        if count % 5 == 0:
            print(f"Processing {count} elements in test set!")
    
    avg_per = total_per / len(features)
    print(f"Average PER: {avg_per:.2f}%")
    print(len(y), len(y[0]))
    if save_output:
        y = np.array(y)
        print(y.shape)
        np.savetxt(OUTPUT_DIR + 'y.csv', y.reshape(y.shape[0], -1).astype(int_type(BIT_WIDTH)), fmt='%i', delimiter=',')

    return avg_per

def sparse_quantized_weights(model):
    dir = OUTPUT_DIR + 'quantized/'
    w_block = model.block_number['w']
    u_block = model.block_number['u']

    if model.what_to_prune['f'][0] == 1:    
        sparsity = model.what_to_prune['f'][1]
        print('\nWriting U_f and W_f\n')

        quantize_matrix(model.wf, 'wf', dir , quantize = False, need_transpose = False)
        model.save_prune_weights(sparsity, 'wf', dir , w_block)
        
        quantize_matrix(model.uf, 'uf', dir , quantize = False, need_transpose = False)
        model.save_prune_weights(sparsity, 'uf', dir , u_block)

    if model.what_to_prune['i'][0] == 1:    
        sparsity = model.what_to_prune['i'][1]
        print('\nWriting U_i and W_i\n')

        quantize_matrix(model.wi, 'wi', dir , quantize = False, need_transpose = False)
        model.save_prune_weights(sparsity, 'wi', dir , w_block)

        quantize_matrix(model.ui, 'ui', dir , quantize = False, need_transpose = False)
        model.save_prune_weights(sparsity, 'ui', dir , u_block)

    if model.what_to_prune['c'][0] == 1:    
        sparsity = model.what_to_prune['c'][1]
        print('\nWriting U_c and W_c\n')
    
        quantize_matrix(model.wc, 'wc', dir , quantize = False, need_transpose = False)
        model.save_prune_weights(sparsity, 'wc', dir , w_block)

        quantize_matrix(model.uc, 'uc', dir , quantize = False, need_transpose = False)
        model.save_prune_weights(sparsity, 'uc', dir , u_block)

    if model.what_to_prune['o'][0] == 1:    
        sparsity = model.what_to_prune['o'][1]
        print('\nWriting U_o and W_o\n')

        quantize_matrix(model.wo, 'wo', dir , quantize = False, need_transpose = False)
        model.save_prune_weights(sparsity, 'wo', dir , w_block)

        quantize_matrix(model.uo, 'uo', dir , quantize = False, need_transpose = False)
        model.save_prune_weights(sparsity, 'uo', dir , u_block)

    quantize_matrix(model.w_proj, 'wp', dir , quantize = False, need_transpose = False)
    if proj_sparsity != 0:
        model.save_prune_weights(proj_sparsity, 'wp', dir , proj_block)


    quantize_matrix(model.wd['1'], 'wd', dir , quantize = False, need_transpose = False)
    quantize_matrix(model.bd['1'].reshape(1, -1), 'bd', dir , quantize = False, need_transpose = False)
    quantize_matrix(model.bf.reshape(1, -1), 'bf', dir , quantize = False, need_transpose = False)
    quantize_matrix(model.bi.reshape(1, -1), 'bi', dir , quantize = False, need_transpose = False)
    quantize_matrix(model.bc.reshape(1, -1), 'bc', dir , quantize = False, need_transpose = False)
    quantize_matrix(model.bo.reshape(1, -1), 'bo', dir , quantize = False, need_transpose = False)


input_size=153
hidden_size=1024
projection_size=512
hidden_units=300
output_size=40
seq_len = 436

what_to_prune={'f':[1,0.95], 'o':[1,0.95], 'i':[1,0.95], 'c':[1,0.95]}
block_numbers={'w': 7, 'u': 25}
proj_sparsity= 0.95
proj_block= 50

# Instantiate your model
model = SparseProjectedLSTM(
    input_size = input_size, 
    hidden_size = hidden_size, 
    projection_size = projection_size,
    hidden_units = hidden_units, 
    output_size = output_size,
    sparsity = True,
    what_to_prune = what_to_prune,
    block_numbers = block_numbers,
    proj_sparsity = proj_sparsity, 
    proj_block = proj_block
)

OUTPUT_DIR = 'timit/'
Final_DIR = 'timit/sparse/'

# To Quantize input before running!!!
# with open(OUTPUT_DIR + "features_20.csv", 'r') as f:
#     reader = csv.reader(f)
#     raw = np.array([[float(v) for v in row] for row in reader])

# print(len(raw), len(raw)/20)
# quantize_input(raw, 153, 'x', OUTPUT_DIR)


# Prepare the projection and weights

model.load_projection(OUTPUT_DIR, quantize= True)
model.load_weights(OUTPUT_DIR, quantize= True, need_transpose = False)
model.load_biases(OUTPUT_DIR, quantize= True)
model.generate_initial_state(OUTPUT_DIR)
model.load_dense_layer(OUTPUT_DIR, quantize= True)
sparse_quantized_weights(model)

model.prune_weights_for_inference()
model.prune_projection()

print("\n\n\n-------Starting Evaluation--------\n\n")

# Evaluate
evaluate_sparse_projected_lstm(model, "x.csv", "labels_20.csv", blank_idx=39, n_input=input_size, seq_len=seq_len, save_output = True)

