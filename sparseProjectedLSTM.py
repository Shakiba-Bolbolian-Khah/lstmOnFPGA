import numpy as np
from fixedPointLStmWeight import *
from sparseLSTM import SparseSHIR_LSTM
from sparseLSTM import *
from fixedPointLStmWeight import SHIR_LSTM


class SparseProjectedLSTM(SparseSHIR_LSTM):
    def __init__(self, input_size, hidden_size, projection_size, hidden_units, output_size,
                 sparsity=False, what_to_prune=None, block_numbers=None, proj_sparsity=0.0, proj_block=1):
        if what_to_prune is None:
            what_to_prune = {'f':[1,0.0], 'o':[1,0.0], 'i':[1,0.0], 'c':[1,0.0]}
        if block_numbers is None:
            block_numbers = {'w': 1, 'u': 1}

        super().__init__(input_size, hidden_size, hidden_units, output_size,
                         sparsity, what_to_prune, block_numbers)
        self.projection_size = projection_size
        self.proj_sparsity = proj_sparsity
        self.proj_block = proj_block

    def load_projection(self, dir="data/", quantize = False, need_transpose = False):
        self.w_proj = load_matrix("wp", dir, quantize, need_transpose)

    def generate_initial_state(self, dir):
        self.h0 = np.zeros((self.projection_size,1), dtype=int_type(BIT_WIDTH))
        self.c0 = np.zeros((self.hidden_size,1), dtype=int_type(BIT_WIDTH))

        np.savetxt(dir + 'state' + '.csv', np.concatenate([self.h0, self.c0]).T.astype(int_type(BIT_WIDTH)), fmt='%i', delimiter=',')


    def prune_projection(self, name="proj", dir="data/"):
        print("Prunning Projection Matrix:\n")
        self.w_proj = self.prune_weights(self.w_proj, self.proj_sparsity, self.proj_block)

    def run_inference(self, item, keep_output=True, test_for_accuracy=False):
        h = self.h0.reshape(self.projection_size,)
        c = self.c0.reshape(self.hidden_size,)
        y = []

        for x_t in item:
            f = sigmoid(fixed_add(fixed_add(fixed_matvec_numpy(self.uf, h), fixed_matvec_numpy(self.wf, x_t)), self.bf))
            i = sigmoid(fixed_add(fixed_add(fixed_matvec_numpy(self.ui, h), fixed_matvec_numpy(self.wi, x_t)), self.bi))
            o = sigmoid(fixed_add(fixed_add(fixed_matvec_numpy(self.uo, h), fixed_matvec_numpy(self.wo, x_t)), self.bo))
            c_prime = tanh(fixed_add(fixed_add(fixed_matvec_numpy(self.uc, h), fixed_matvec_numpy(self.wc, x_t)), self.bc))

            c = fixed_add(fixed_mul(f, c), fixed_mul(i, c_prime))
            h = fixed_mul(tanh(c), o)

            h = fixed_matvec_numpy(self.w_proj, h)  # Projected hidden state

            output = h if test_for_accuracy else np.concatenate([h, c])
            if keep_output:
                y += [output]
            else:
                y = output

        y = np.array(y)
        return y

    def run_LSTM(self, dir, input, is_input_file=True, test_for_accuracy=False, dense_activation="none"):
        # self.load_weights(dir)
        # self.load_biases(dir)
        # self.generate_initial_state(dir)
        # self.load_dense_layer(dir)
        # self.load_projection(dir)

        # if self.sparsity:
        #     self.prune_weights_for_inference()
        # if self.proj_sparsity > 0:
        #     self.prune_projection()

        # if is_input_file:
        #     self.load_input(dir)
        # else:
        #     self.x = np.round(input * (2**FRAC_BITS)).astype(int_type(BIT_WIDTH))
        y = []
        

        lstm_out = self.run_inference(input, keep_output=True, test_for_accuracy=True)
        lstm_out = lstm_out.reshape(-1, self.projection_size)

        for item in lstm_out:
            y_item = self.run_dense(item, activation=dense_activation)
            y.append(y_item)

        y = np.array(y)
        if is_input_file:
            np.savetxt(dir + 'y.csv', y.astype(int_type(BIT_WIDTH)), fmt='%i', delimiter=',')
        return y
