import numpy as np
from fixedPointLStmWeight import *


def prune_matrix_by_row_percent(matrix: np.ndarray, percent: float) -> np.ndarray:
    """
    Prunes each row of the matrix by removing a fixed percentage of the smallest absolute values.
    The output has the same shape as the input.
    """
    pruned_matrix = np.copy(matrix)
    print("Not-Pruned Matrix: " , matrix.shape, '  Num of Zero in first row:  ', np.sum(matrix[0]==0))

    for i in range(matrix.shape[0]):
        row = matrix[i]
        abs_row = np.abs(row)
        n_keep = int(np.ceil((1 - percent) * len(row)))

        if n_keep == 0:
            top_indices = np.argsort(-abs_row)[:1]  # keep at least one element
        else:
            top_indices = np.argsort(-abs_row)[:n_keep]

        mask = np.zeros_like(row, dtype=np.int8)
        mask[top_indices] = 1
        pruned_matrix[i] = row * mask

    sparsity_level = 1 - np.count_nonzero(pruned_matrix) / pruned_matrix.size
    print("Sparsity Level: ", sparsity_level)
    print("Pruned Matrix:    " , pruned_matrix.shape, '  Num of Zero in first row:  ', np.sum(pruned_matrix[0] == 0))

    return pruned_matrix


class SparseSHIR_LSTM(SHIR_LSTM):
    def __init__(self, input_size, hidden_size, hidden_units, output_size, sparsity= 0.0, what_to_prune= {'f':[1,0.0], 'o':[1,0.0], 'i':[1,0.0], 'c':[1,0.0]}):
        super().__init__(input_size, hidden_size, hidden_units, output_size)
        self.sparsity = sparsity
        self.what_to_prune = what_to_prune

    def prune_weight_matrix(self, sparsity, file_name, dir):
        """
        Prunes a weight matrix row-wise by keeping the top-N% absolute values.
        Outputs value and index matrices and saves them as CSV files.

        Parameters:
            weight_matrix (str): Path to the input CSV weight matrix.
            top_percent (float): Fraction of values to keep (e.g., 0.2 for 20%).
            dir (str): Directory to save the output CSV files.
        """
        top_percent = 1- sparsity
        weight_matrix = dir + file_name + '.csv'
        # Load matrix
        matrix = np.loadtxt(weight_matrix, delimiter=",")
        num_rows, num_cols = matrix.shape
        n_keep = int(np.ceil(top_percent * num_cols))

        value_matrix = []
        index_matrix = []

        for row in matrix:
            abs_row = np.abs(row)
            top_indices = np.argpartition(-abs_row, n_keep)[:n_keep]
            sorted_indices = np.sort(top_indices)  # sort indices ascending
            pruned_values = row[sorted_indices]

            value_matrix.append(pruned_values)
            index_matrix.append(sorted_indices)

        value_matrix = np.array(value_matrix)
        index_matrix = np.array(index_matrix, dtype=int)

        np.savetxt(dir+'sparse/'+file_name+"_pruned_values.csv", value_matrix, delimiter=",", fmt="%i")
        np.savetxt(dir+'sparse/'+file_name+"_indices.csv", index_matrix, delimiter=",", fmt="%d")

        print(f"Saved pruned values and indices for {file_name} with shape of {matrix.shape} to {dir} with shape of {value_matrix.shape}")


    def prune_weights_for_inference(self):
        if self.what_to_prune['f'][0] == 1:    
            sparsity = self.what_to_prune['f'][1]
            self.uf = prune_matrix_by_row_percent(self.uf, sparsity)
            self.wf = prune_matrix_by_row_percent(self.wf, sparsity)
        
        if self.what_to_prune['i'][0] == 1:
            sparsity = self.what_to_prune['i'][1]
            self.ui = prune_matrix_by_row_percent(self.ui, sparsity)
            self.wi = prune_matrix_by_row_percent(self.wi, sparsity)

        if self.what_to_prune['o'][0] == 1:
            sparsity = self.what_to_prune['o'][1]
            self.uo = prune_matrix_by_row_percent(self.uo, sparsity)
            self.wo = prune_matrix_by_row_percent(self.wo, sparsity)

        if self.what_to_prune['c'][0] == 1:
            sparsity = self.what_to_prune['c'][1]
            self.uc = prune_matrix_by_row_percent(self.uc, sparsity)
            self.wc = prune_matrix_by_row_percent(self.wc, sparsity)

    def save_pruned_weights(self, dir):
        if self.what_to_prune['f'][0] == 1:    
            sparsity = self.what_to_prune['f'][1]
            self.prune_weight_matrix(sparsity, 'wf', dir)
            self.prune_weight_matrix(sparsity, 'uf', dir)
        
        if self.what_to_prune['i'][0] == 1:
            sparsity = self.what_to_prune['i'][1]
            self.prune_weight_matrix(sparsity, 'wi', dir)
            self.prune_weight_matrix(sparsity, 'ui', dir)

        if self.what_to_prune['o'][0] == 1:
            sparsity = self.what_to_prune['o'][1]
            self.prune_weight_matrix(sparsity, 'wo', dir)
            self.prune_weight_matrix(sparsity, 'uo', dir)

        if self.what_to_prune['c'][0] == 1:
            sparsity = self.what_to_prune['c'][1]
            self.prune_weight_matrix(sparsity, 'wc', dir)
            self.prune_weight_matrix(sparsity, 'uc', dir)


    def run_LSTM(self, dir, input, is_input_file=True, test_for_accuracy=False, dense_activation="none"):
        self.load_weights(dir)
        self.load_biases(dir)
        self.generate_initial_state(dir)
        self.load_dense_layer(dir)
        print("Activation: ",dense_activation)

        if self.sparsity > 0:
            self.prune_weights_for_inference()

        if is_input_file:
            self.load_input(dir)
        else:
            self.x = np.round(input * (2**FRAC_BITS)).astype(int_type(BIT_WIDTH))

        y = []
        for item in self.x:
            y_item = self.run_dense(self.run_inference(item, False, True), num='1', activation=dense_activation)
            y += [y_item]

        y = np.array(y)
        if is_input_file:
            np.savetxt(dir + 'y.csv', y.astype(int_type(BIT_WIDTH)), fmt='%i', delimiter=',')
        return y
