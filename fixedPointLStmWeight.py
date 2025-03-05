import numpy as np

FRAC_BITS = 3
BIT_WIDTH = 16
MIN = -4
MAX = 8
PAR_DEG = 1

MAX_INT = 2**(BIT_WIDTH-1)
MIN_INT = -(2**(BIT_WIDTH-1))
DIR = 'data/'


def bankers_round(value: int, low_elements: int = FRAC_BITS) -> int:
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


def tanh(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, -(2**FRAC_BITS), 2**FRAC_BITS)

def sigmoid(x, scale = 2**FRAC_BITS):
    
    SHIFT_AMOUNT = 2
    ONE = scale
    OFFSET = scale // 2  # Represents 0.5

    # Perform an arithmetic right shift (for both scalars and numpy arrays)
    # Python's ">>" operator works elementwise on numpy arrays of int type.
    x_div4 = x >> SHIFT_AMOUNT

    # Add OFFSET (0.5 in fixed-point)
    y = x_div4 + OFFSET

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
 
    # Convert to int32 to prevent overflow during addition
    result = a.astype(int_type(BIT_WIDTH*2)) + b.astype(int_type(BIT_WIDTH*2))
    result = v_bankers_round(result, 0)
    
    # result = np.clip(result, MIN_INT, MAX_INT).astype(int_type(BIT_WIDTH))
    return result.astype(int_type(BIT_WIDTH))


def fixed_mul(a: np.ndarray, b: np.ndarray, frac_bits: int = FRAC_BITS) -> np.ndarray:
    # Multiply as int32 to prevent overflow
    product = a.astype(np.int32) * b.astype(int_type(BIT_WIDTH*2))
    # product = product / (2 ** frac_bits)
    product = v_bankers_round(product) #np.round(product)

    # product = np.clip(product, MIN_INT, MAX_INT).astype(int_type(BIT_WIDTH))
    return product.astype(int_type(BIT_WIDTH))


def fixed_matvec(matrix: np.ndarray, vector: np.ndarray, frac_bits: int = FRAC_BITS, low_elements: int = 2) -> np.ndarray:

    m, n = matrix.shape

    products = np.empty((m, n), dtype=np.int32)
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
    np.savetxt(DIR + name + '.csv', vector.astype(int_type(BIT_WIDTH)), fmt='%i', delimiter=',')
    
    return vector



class LSTM:
    def __init__(self, input_size, hidden_size, hidden_units):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_units = hidden_units

    def generate_weights(self):
        self.uf = generate_random_matrix(self.hidden_size, self.hidden_size, "uf")
        self.ui = generate_random_matrix(self.hidden_size, self.hidden_size, "ui")
        self.uo = generate_random_matrix(self.hidden_size, self.hidden_size, "uo")
        self.uc = generate_random_matrix(self.hidden_size, self.hidden_size, "uc")

        self.wf = generate_random_matrix(self.hidden_size, self.input_size, "wf")
        self.wi = generate_random_matrix(self.hidden_size, self.input_size, "wi")
        self.wo = generate_random_matrix(self.hidden_size, self.input_size, "wo")
        self.wc = generate_random_matrix(self.hidden_size, self.input_size, "wc")

    def generate_biases(self):
        self.bf = generate_random_vector(self.hidden_size, "bf")
        self.bi = generate_random_vector(self.hidden_size, "bi")
        self.bo = generate_random_vector(self.hidden_size, "bo")
        self.bc = generate_random_vector(self.hidden_size, "bc")

    def generate_initial_state(self):
        self.h0 = generate_random_vector(self.hidden_size, "h0")
        self.c0 = generate_random_vector(self.hidden_size, "c0")

        np.savetxt(DIR + 'state' + '.csv', np.concatenate([self.h0, self.c0]).T.astype(int_type(BIT_WIDTH)), fmt='%i', delimiter=',')

    def generate_input(self):
        self.x = generate_random_matrix(self.hidden_units, self.input_size, "x", MIN*2, MAX*2, 1)


    def run_test(self):
        y = []

        for x_t in self.x:
            f = fixed_matvec(self.wf, x_t)
            y+= [f]

        y = np.array(y)
        print(y)
        np.savetxt(DIR + 'y.csv', y.astype(int_type(BIT_WIDTH)), fmt='%i', delimiter=',')



    def run_inference(self):
        h = self.h0.reshape(self.hidden_size,)
        c = self.c0.reshape(self.hidden_size,)
        y = []

        for x_t in self.x:

            f = sigmoid(fixed_matvec(self.uf, h)+ fixed_matvec(self.wf, x_t))

            i = sigmoid(fixed_matvec(self.ui, h)+ fixed_matvec(self.wi, x_t))

            o = sigmoid(fixed_matvec(self.uo, h)+ fixed_matvec(self.wo, x_t))

            c_prime = tanh(fixed_matvec(self.uc, h)+ fixed_matvec(self.wc, x_t))

            c = fixed_add(fixed_mul(f, c),fixed_mul(i, c_prime))

            h = fixed_mul(tanh(c),o)

            print("c,", c.astype(np.int16))
            print("h,",h.astype(np.int16))

            y += [np.concatenate([h,c])]

        y = np.array(y)
        np.savetxt(DIR + 'y.csv', y.astype(int_type(BIT_WIDTH)), fmt='%i', delimiter=',')

    def run_LSTM(self):
        self.generate_weights()
        self.generate_biases()
        self.generate_initial_state()
        self.generate_input()
        self.run_inference()


lstm = LSTM(8,16,8)
lstm.run_LSTM()

