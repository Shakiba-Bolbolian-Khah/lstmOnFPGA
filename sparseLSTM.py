import numpy as np
from fixedPointLStmWeight import *
import os

# def prune_matrix_by_block_percent(matrix: np.ndarray, percent: float, n_blocks: int) -> np.ndarray:
#     """
#     Prunes each row of the matrix by dividing it into blocks and removing a percentage of the 
#     smallest absolute values within each block.

#     Parameters:
#         matrix (np.ndarray): The matrix to prune
#         percent (float): Percentage of values to remove in each block
#         n_blocks (int): Number of blocks to divide each row into
#     """
#     pruned_matrix = np.copy(matrix)
#     rows, cols = matrix.shape
#     # assert cols % n_blocks == 0, "Number of columns must be divisible by number of blocks"
#     block_size = cols // n_blocks

#     for i in range(rows):
#         for b in range(n_blocks):
#             start = b * block_size
#             end = (b + 1) * block_size
#             block = matrix[i, start:end]
#             abs_block = np.abs(block)

#             val = (1 - percent) * block_size
#             if val > 1 and (val - int(val)) < 0.5:
#                 n_keep = int(np.floor(val))
#             else:
#                 n_keep = int(np.ceil(val))
#             if n_keep == 0:
#                 top_indices = np.argsort(-abs_block)[:1]
#             else:
#                 top_indices = np.argsort(-abs_block)[:n_keep]

#             mask = np.zeros_like(block, dtype=np.int8)
#             mask[top_indices] = 1
#             pruned_matrix[i, start:end] = block * mask

#     sparsity_level = 1 - np.count_nonzero(pruned_matrix) / pruned_matrix.size
#     print("Block-wise Sparsity Level: ", sparsity_level)
#     print("Pruned Matrix:    " , pruned_matrix.shape, '  Num of Zero in first row:  ', np.sum(pruned_matrix[0] == 0))
#     return pruned_matrix

def prune_matrix_by_block_percent(matrix: np.ndarray, percent: float, n_blocks: int) -> np.ndarray:
    """
    Prunes each row of the matrix by dividing it into blocks and removing a percentage of the 
    smallest absolute values within each block (block-wise pruning).
    
    Parameters:
        matrix (np.ndarray): The matrix to prune
        percent (float): Percentage of values to remove in each block (0.0 to 1.0)
        n_blocks (int): Number of blocks to divide each row into
    """
    pruned_matrix = np.copy(matrix)
    rows, cols = matrix.shape
    block_size = int(np.ceil(cols / n_blocks))  # Match second function

    top_percent = 1 - percent

    for i in range(rows):
        for b in range(n_blocks):
            start = b * block_size
            end = min((b + 1) * block_size, cols)
            block = matrix[i, start:end]
            block_length = end - start
            if block_length == 0:
                continue

            val = top_percent * block_length
            if  (val - int(val)) < 0.5: #val > 1 and
                n_keep = int(np.floor(val))
            else:
                n_keep = int(np.ceil(val))

            abs_block = np.abs(block)
            if n_keep == 0:
                top_indices = np.argsort(-abs_block)[:1]  # keep at least one
            else:
                # Fast top-k using argpartition, like second function
                top_indices = np.argpartition(-abs_block, n_keep - 1)[:n_keep]

            mask = np.zeros_like(block, dtype=np.int8)
            mask[top_indices] = 1
            pruned_matrix[i, start:end] = block * mask

    sparsity_level = 1 - np.count_nonzero(pruned_matrix) / pruned_matrix.size
    print("Block-wise Sparsity Level: ", sparsity_level)
    print("Pruned Matrix:    ", pruned_matrix.shape, '  Num of non-Zero in first row:  ', len(pruned_matrix[0])-(np.sum(pruned_matrix[0] == 0)))
    return pruned_matrix


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
    def __init__(self, input_size, hidden_size, hidden_units, output_size, sparsity= False, 
                 what_to_prune= {'f':[1,0.0], 'o':[1,0.0], 'i':[1,0.0], 'c':[1,0.0]}, 
                 block_numbers= {'w': 1, 'u': 1}):

        super().__init__(input_size, hidden_size, hidden_units, output_size)
        self.sparsity = sparsity
        self.what_to_prune = what_to_prune
        self.block_number = block_numbers

    def prune_save_weight_matrix_block(self, sparsity, file_name, dir, block_number):
        """
        Performs block row-wise pruning on a weight matrix.

        For each row:
        - Divides it into `block_number` blocks.
        - In each block, removes the lowest `sparsity` fraction (by absolute value).
        - Stores the remaining values and their indices relative to the block (starting at 0).
        
        Parameters:
            sparsity (float): Fraction of values to remove in each block (0.0 to 1.0).
            file_name (str): Name of the input CSV (without extension).
            dir (str): Directory of input and where output is saved.
            block_number (int): Number of blocks per row.
        """
        top_percent = 1 - sparsity
        weight_matrix_path = dir + file_name + '.csv'
        
        matrix = np.loadtxt(weight_matrix_path, delimiter=",")
        num_rows, num_cols = matrix.shape
        block_size = int(np.ceil(num_cols / block_number))

        value_matrix = []
        index_matrix = []

        for row in matrix:
            row_values = []
            row_indices = []

            for b in range(block_number):
                start = b * block_size
                end = min((b + 1) * block_size, num_cols)
                block = row[start:end]
                block_length = end - start
                if block_length == 0:
                    continue

                val = top_percent * block_length
                if (val - int(val)) < 0.5: #val > 1 and 
                    n_keep = int(np.floor(val))
                else:
                    n_keep = int(np.ceil(val))

                abs_block = np.abs(block)
                top_indices = np.argpartition(-abs_block, n_keep - 1)[:n_keep]
                sorted_indices = np.sort(top_indices)
                pruned_vals = block[sorted_indices]
                relative_indices = sorted_indices  # Already relative to the block

                row_values.append(pruned_vals)
                row_indices.append(relative_indices)

            if row_values:
                flat_values = np.concatenate(row_values)
                flat_indices = np.concatenate(row_indices)
            else:
                flat_values = np.array([])
                flat_indices = np.array([], dtype=int)

            value_matrix.append(flat_values)
            index_matrix.append(flat_indices)

        # Padding rows to the same length
        max_len = max(len(row) for row in value_matrix)

        value_matrix_pad = np.array([
            np.pad(row, (0, max_len - len(row)), mode='constant', constant_values=0)
            for row in value_matrix
        ])
        index_matrix_pad = np.array([
            np.pad(row, (0, max_len - len(row)), mode='constant', constant_values=-1)
            for row in index_matrix
        ])

        sparse_dir = os.path.join(dir, 'sparse')
        os.makedirs(sparse_dir, exist_ok=True)

        np.savetxt(dir+'sparse/'+file_name + "_pruned_values_block.csv",
                value_matrix_pad, delimiter=",", fmt="%i")
        np.savetxt(dir+'sparse/'+file_name + "_indices_block.csv",
                index_matrix_pad, delimiter=",", fmt="%d")

        print(f"Saved pruned values and block-relative indices for {file_name}")
        print(f"Original shape: {matrix.shape}, Block size: {block_size}, Padded shape: {value_matrix_pad.shape}")

    def prune_save_weight_matrix(self, sparsity, matrix):
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

    def prune_weights(self, weight, sparsity, block=1):
        if block == 1:
            return prune_matrix_by_row_percent(weight, sparsity)
        else:
            return prune_matrix_by_block_percent(weight, sparsity,  block)
        
    def save_prune_weights(self, sparsity, file_name, dir, block=1):
        if block == 1:
            return self.prune_save_weight_matrix(sparsity, file_name, dir)
        else:
            return self.prune_save_weight_matrix_block(sparsity, file_name, dir, block)

    def prune_weights_for_inference(self):
        if self.what_to_prune['f'][0] == 1:    
            sparsity = self.what_to_prune['f'][1]
            print('U_f and W_f\n')
            self.uf = self.prune_weights(self.uf, sparsity, self.block_number['u'])
            self.wf = self.prune_weights(self.wf, sparsity, self.block_number['w'])
        
        if self.what_to_prune['i'][0] == 1:
            sparsity = self.what_to_prune['i'][1]
            print('U_i and W_i\n')
            self.ui = self.prune_weights(self.ui, sparsity, self.block_number['u'])
            self.wi = self.prune_weights(self.wi, sparsity, self.block_number['w'])

        if self.what_to_prune['o'][0] == 1:
            sparsity = self.what_to_prune['o'][1]
            print('U_o and W_o\n')
            self.uo = self.prune_weights(self.uo, sparsity, self.block_number['u'])
            self.wo = self.prune_weights(self.wo, sparsity, self.block_number['w'])

        if self.what_to_prune['c'][0] == 1:
            print('U_c and W_c\n')
            sparsity = self.what_to_prune['c'][1]
            self.uc = self.prune_weights(self.uc, sparsity, self.block_number['u'])
            self.wc = self.prune_weights(self.wc, sparsity, self.block_number['w'])

    def save_pruned_weights(self, dir):
        if self.what_to_prune['f'][0] == 1:    
            sparsity = self.what_to_prune['f'][1]
            self.save_prune_weights(sparsity, 'wf', dir, self.block_number['w'])
            self.save_prune_weights(sparsity, 'uf', dir, self.block_number['u'])
        
        if self.what_to_prune['i'][0] == 1:
            sparsity = self.what_to_prune['i'][1]
            self.save_prune_weights(sparsity, 'wi', dir, self.block_number['w'])
            self.save_prune_weights(sparsity, 'ui', dir, self.block_number['u'])

        if self.what_to_prune['o'][0] == 1:
            sparsity = self.what_to_prune['o'][1]
            self.save_prune_weights(sparsity, 'wo', dir, self.block_number['w'])
            self.save_prune_weights(sparsity, 'uo', dir, self.block_number['u'])

        if self.what_to_prune['c'][0] == 1:
            sparsity = self.what_to_prune['c'][1]
            self.save_prune_weights(sparsity, 'wc', dir, self.block_number['w'])
            self.save_prune_weights(sparsity, 'uc', dir, self.block_number['u'])


    def run_LSTM(self, dir, input, is_input_file=True, test_for_accuracy=False, dense_activation="none"):
        self.load_weights(dir)
        self.load_biases(dir)
        self.generate_initial_state(dir)
        self.load_dense_layer(dir)
        print("Activation: ",dense_activation)

        if self.sparsity:
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
