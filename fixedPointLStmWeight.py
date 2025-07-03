import numpy as np
import csv

FRAC_BITS = 10
BIT_WIDTH = 16
MIN = -8
MAX = 16
PAR_DEG = 1

MAX_INT = 2**(BIT_WIDTH-1)
MIN_INT = -(2**(BIT_WIDTH-1))
DIR = 'data/'


def bankers_round(value, low_elements: int = FRAC_BITS) -> int:
    """
    Mimics the VHDL rounding which appends "00" to the fractional bits.
    
    The fixed-point number is assumed to have the lower 'low_elements' bits as
    the fractional portion. The VHDL code concatenates "00" to these bits,
    so we simulate that by shifting left by 2.
    """

    # Get the integer part.
    round_hi = value >> low_elements
    # Extract the fractional part and "append" two zeros (shift left by 2). << 2 is additional to mimic the VHDL design
    fractional = (value & ((1 << low_elements) - 1)) << 2
    # The new number of fractional bits is low_elements + 2.
    new_low_elements = low_elements + 2
    # The threshold for exactly 0.5 in this extended fraction.
    threshold = 1 << (new_low_elements - 1)
    
    if fractional > threshold:
        return round_hi + 1 if value >= 0 else round_hi - 1
    elif fractional < threshold:
        return round_hi
    else:
        # If exactly half, round to even.
        if (round_hi & 1) != 0:  # round_hi is odd.
            return round_hi + 1 if value >= 0 else round_hi - 1
        else:
            return round_hi

v_bankers_round = np.vectorize(bankers_round)


def tanh_approx(x):
    if x <= -1.5:
        return -1
    elif -1.5 < x <= -0.5:
        return x + 0.5
    elif -0.5 < x <= 0.5:
        return x
    elif 0.5 < x <= 1.5:
        return x - 0.5
    else:
        return 1


v_tanh = np.vectorize(tanh_approx)


def tanh(x, scale = 2**FRAC_BITS):
    # return (v_tanh(x/scale)*(scale)).astype(int_type(BIT_WIDTH))
    # return (np.tanh(x/(scale))*(scale)).astype(int_type(BIT_WIDTH))
    return np.clip(x, -(scale), scale)


def sigmoid_approx(x):
    # return 1/(1 + np.exp(-x)) 
    
    if x <= -4:
        return 0.0
    elif x <= -2:
        return 0.0505 * (x + 4) + 0.018
    elif x <= 0:
        return 0.1905 * (x + 2) + 0.119
    elif x <= 2:
        return 0.1905 * x + 0.5
    elif x <= 4:
        return 0.0505 * (x - 2) + 0.881
    else:
        return 1.0
    

v_sigmoid = np.vectorize(sigmoid_approx)


def sigmoid(x, scale = 2**FRAC_BITS):
    
    return (v_sigmoid(x/scale)*(scale)).astype(int_type(BIT_WIDTH))
    SHIFT_AMOUNT = 2
    ONE = scale
    OFFSET = scale // 2  # Represents 0.5

    # Perform an arithmetic right shift (for both scalars and numpy arrays)
    # Python's ">>" operator works elementwise on numpy arrays of int type.
    x_div4 = x >> SHIFT_AMOUNT

    # Add OFFSET (0.5 in fixed-point)
    y = (x_div4 + OFFSET).astype(int_type(BIT_WIDTH))

    y_max = np.where(y > 0, y, 0)
    y_clip = np.where(y_max < ONE, y_max, ONE)

    return y_clip


def int_type(N: int):
    try:
        # Construct the dtype string (e.g. 'int16')
        dtype_str = f'int{N}'
        return np.dtype(dtype_str).type
    except TypeError:
        raise ValueError(f"Unsupported integer width: {N}")


def fixed_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
 
    result = a.astype(int_type(BIT_WIDTH*2)) + b.astype(int_type(BIT_WIDTH*2))
    # result = v_bankers_round(result, 0)
    
    # result = np.clip(result, MIN_INT, MAX_INT).astype(int_type(BIT_WIDTH))
    return result.astype(int_type(BIT_WIDTH))


def fixed_mul(a: np.ndarray, b: np.ndarray, frac_bits: int = FRAC_BITS) -> np.ndarray:

    product = a.astype(int_type(BIT_WIDTH*2)) * b.astype(int_type(BIT_WIDTH*2))
    # product = product / (2 ** frac_bits)
    product = v_bankers_round(product) 

    return product.astype(int_type(BIT_WIDTH))


def fixed_matvec(matrix: np.ndarray, vector: np.ndarray, frac_bits: int = FRAC_BITS, low_elements: int = 2) -> np.ndarray:

    m, n = matrix.shape

    products = np.empty((m, n), dtype=int_type(BIT_WIDTH*2))
    for i in range(m):
        for j in range(n):

            prod = int(matrix[i, j]) * int(vector[j])
            products[i, j] = bankers_round(prod)
    
    acc = np.sum(products, axis=1)
    result = v_bankers_round(acc, 0)
    return result.astype(int_type(BIT_WIDTH))


def fixed_matvec_numpy(matrix: np.ndarray, vector: np.ndarray, frac_bits: int = FRAC_BITS) -> np.ndarray:
    acc = matrix.astype(int_type(BIT_WIDTH*2)) @ vector.astype(int_type(BIT_WIDTH*2))
    result = v_bankers_round(acc)

    return result.astype(int_type(BIT_WIDTH))


def generate_random_matrix(row, column, name: str, min_value: int = MIN, max_value: int = MAX, par_deg: int = PAR_DEG) -> np.ndarray :
    
    matrix = np.random.randint(min_value, max_value, size = (row, column))
    np.savetxt(DIR + name + '.csv', matrix.astype(int_type(BIT_WIDTH)), fmt='%i', delimiter=',')

    if par_deg != 1:
        reshaped_matrix = np.reshape(matrix, (row // par_deg, column * par_deg))
        np.savetxt(DIR + name + str(PAR_DEG)+ '.csv', reshaped_matrix.astype(int_type(BIT_WIDTH)), fmt='%i', delimiter=',')

    return matrix


def generate_random_vector(length, name: str, min_value: int = MIN, max_value: int = MAX) -> np.ndarray :
    vector = np.random.randint(min_value, max_value, size=(length, 1))
    np.savetxt(DIR + name + '.csv', vector.astype(int_type(BIT_WIDTH)).T, fmt='%i', delimiter=',')
    
    return vector


def quantize_matrix(data, name: str, dir: str, quantize: bool = True, need_transpose: bool = True):

    reshaped_data = data.T if need_transpose else data

    if quantize:
        quantized_data = np.round(reshaped_data * (2**FRAC_BITS)).astype(int_type(BIT_WIDTH))
        np.savetxt(dir + name + '.csv', quantized_data.astype(int_type(BIT_WIDTH)), fmt='%i', delimiter=',')
    else:
        np.savetxt(dir + name + '.csv', reshaped_data.astype(int_type(BIT_WIDTH)), fmt='%i', delimiter=',')


def quantize_input(data, n_input, name: str, dir: str, quantize: bool = True):
    
    reshaped_data = data.reshape(-1, n_input)
    if quantize:
        quantized_data = np.round(reshaped_data * (2**FRAC_BITS)).astype(int_type(BIT_WIDTH))
        np.savetxt(dir + name + '.csv', quantized_data.astype(int_type(BIT_WIDTH)), fmt='%i', delimiter=',')
    else:
        np.savetxt(dir + name + '.csv', reshaped_data.astype(np.float32), fmt='%f', delimiter=',')


def load_matrix(name: str, dir: str, quantize: bool = False, need_transpose: bool = False):

    with open(dir + name +'.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    if quantize:
        data_array = np.array(data, dtype=np.float32)
        data_array = v_bankers_round((data_array * (2**FRAC_BITS)).astype(int_type(BIT_WIDTH)),0)
    else:
        data_array = np.array(data, dtype=int_type(BIT_WIDTH))
        
    data_array = data_array.T if need_transpose else data_array
    return data_array


class SHIR_LSTM:
    def __init__(self, input_size, hidden_size, hidden_units, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_units = hidden_units
        self.output_size = output_size

    def generate_weights(self):
        self.uf = generate_random_matrix(self.hidden_size, self.hidden_size, "uf")
        self.ui = generate_random_matrix(self.hidden_size, self.hidden_size, "ui")
        self.uo = generate_random_matrix(self.hidden_size, self.hidden_size, "uo")
        self.uc = generate_random_matrix(self.hidden_size, self.hidden_size, "uc")

        self.wf = generate_random_matrix(self.hidden_size, self.input_size, "wf")
        self.wi = generate_random_matrix(self.hidden_size, self.input_size, "wi")
        self.wo = generate_random_matrix(self.hidden_size, self.input_size, "wo")
        self.wc = generate_random_matrix(self.hidden_size, self.input_size, "wc")


    def load_weights(self, dir, quantize = False, need_transpose = False):
        self.uf = load_matrix("uf", dir, quantize, need_transpose)
        self.ui = load_matrix("ui", dir, quantize, need_transpose)
        self.uo = load_matrix("uo", dir, quantize, need_transpose)
        self.uc = load_matrix("uc", dir, quantize, need_transpose)

        self.wf = load_matrix("wf", dir, quantize, need_transpose)
        self.wi = load_matrix("wi", dir, quantize, need_transpose)
        self.wo = load_matrix("wo", dir, quantize, need_transpose)
        self.wc = load_matrix("wc", dir, quantize, need_transpose)


    def generate_biases(self):
        self.bf = generate_random_vector(self.hidden_size, "bf").reshape(self.hidden_size,)
        self.bi = generate_random_vector(self.hidden_size, "bi").reshape(self.hidden_size,)
        self.bo = generate_random_vector(self.hidden_size, "bo").reshape(self.hidden_size,)
        self.bc = generate_random_vector(self.hidden_size, "bc").reshape(self.hidden_size,)


    def load_biases(self, dir, quantize = False):
        self.bf = load_matrix("bf", dir, quantize).reshape(self.hidden_size,)
        self.bi = load_matrix("bi", dir, quantize).reshape(self.hidden_size,)
        self.bo = load_matrix("bo", dir, quantize).reshape(self.hidden_size,)
        self.bc = load_matrix("bc", dir, quantize).reshape(self.hidden_size,)


    def generate_initial_state(self, dir):
        self.h0 = np.zeros((self.hidden_size,1), dtype=int_type(BIT_WIDTH))
        self.c0 = np.zeros((self.hidden_size,1), dtype=int_type(BIT_WIDTH))

        np.savetxt(dir + 'state' + '.csv', np.concatenate([self.h0, self.c0]).T.astype(int_type(BIT_WIDTH)), fmt='%i', delimiter=',')

    def load_dense_layer(self, dir, num = 1, quantize = False, need_transpose = False):
        self.wd = {}
        self.bd = {}
        if num != 1:
            for i in range(num):
                new_name = str(i + 1)

                self.wd[new_name] = load_matrix("wd"+new_name, dir, quantize, need_transpose)
                self.bd[new_name] = load_matrix("bd"+new_name, dir, quantize).reshape(-1,)
        else:
            new_name = '1'
            self.wd[new_name] = load_matrix("wd", dir, quantize, need_transpose)
            self.bd[new_name] = load_matrix("bd", dir, quantize).reshape(-1,)


    def generate_input(self):
        self.x = generate_random_matrix(self.hidden_units, self.input_size, "x", MIN, MAX, 1)


    def load_input(self, dir):
        data_array = load_matrix("x", dir).reshape(-1, self.hidden_units, self.input_size)
        self.x = data_array


    def run_inference(self, item, keep_output = True, test_for_accuracy = False):
        h = self.h0.reshape(self.hidden_size,)
        c = self.c0.reshape(self.hidden_size,)
        y = []

        for x_t in item:

            f = sigmoid(fixed_add(fixed_add(fixed_matvec_numpy(self.uf, h), fixed_matvec_numpy(self.wf, x_t)),self.bf))

            i = sigmoid(fixed_add(fixed_add(fixed_matvec_numpy(self.ui, h), fixed_matvec_numpy(self.wi, x_t)), self.bi))

            o = sigmoid(fixed_add(fixed_add(fixed_matvec_numpy(self.uo, h), fixed_matvec_numpy(self.wo, x_t)), self.bo))

            c_prime = tanh(fixed_add(fixed_add(fixed_matvec_numpy(self.uc, h), fixed_matvec_numpy(self.wc, x_t)),self.bc))

            c = fixed_add(fixed_mul(f, c),fixed_mul(i, c_prime))

            h = fixed_mul(tanh(c),o)

            output = h if test_for_accuracy else np.concatenate([h,c])
            if keep_output:
                y += [output]
            else:
                y = output

        y = np.array(y)
        # np.savetxt(DIR + 'y.csv', y.astype(int_type(BIT_WIDTH)), fmt='%i', delimiter=',')
        return y

    def run_dense(self, item, num = '1', activation = "none"):
        output = fixed_add(fixed_matvec_numpy(self.wd[num], item), self.bd[num])
        if activation == "none":
            return output
        elif activation =="sigmoid":
            return sigmoid(output)

    def run_LSTM(self, dir, input, is_input_file = True, test_for_accuracy = False, dense_activation = "none"):
        self.load_weights(dir)
        self.load_biases(dir)
        self.generate_initial_state(dir)
        self.load_dense_layer(dir)
        if is_input_file:
            self.load_input(dir)
        else:
            # reshaped_data = input.reshape(-1, self.input_size)
            self.x = np.round(input * (2**FRAC_BITS)).astype(int_type(BIT_WIDTH))

        print(self.x.shape)

        y = []
        for item in self.x:
            y_item = self.run_dense(self.run_inference(item, False, True), dense_activation)
            y+= [y_item]
        
        print(type(y[0]))
        y = np.array(y)
        if is_input_file:
            np.savetxt(dir + 'y.csv', y.astype(int_type(BIT_WIDTH)), fmt='%i', delimiter=',')
        return(y)
    
    def run_two_dense(self, dir, test_for_accuracy = False):
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