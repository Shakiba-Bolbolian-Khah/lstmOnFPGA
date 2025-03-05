
import numpy as np
import csv


hiddenSize = 4
inputSize = 4
outputSize = 4
length = 3
seqLength = 1
parDeg = 2
min = 0
max = 2
fileLoc = 'data/'
fraction = 3
quantization = 2**fraction
bitwidth = 16

MAX_INT = 2**(bitwidth-1)
MIN_INT = -(2**(bitwidth-1))

# tanh = lambda x : min(float(quantization), max(float(-1*quantization), x))

def float_to_fixed16(arr: np.ndarray, frac_bits: int = 15) -> np.ndarray:
    """
    Convert a numpy array of floats to 16-bit fixed-point representation (Q15 format).
    
    Parameters:
      arr: numpy array of floating-point numbers.
      frac_bits: number of fractional bits (default is 15 for Q15 format).
      
    Returns:
      numpy array of type int16 containing the fixed-point representation.
    """
    scale = 2 ** frac_bits
    fixed = np.round(arr * scale).astype(np.int32)  # Use int32 to prevent overflow during rounding
    # Clip to 16-bit signed integer range
    fixed = np.clip(fixed, MIN_INT, MAX_INT).astype(np.int16)
    
    return fixed


def fixed16_to_float(arr: np.ndarray, frac_bits: int = fraction) -> np.ndarray:
    """
    Convert a numpy array of 16-bit fixed-point (Q15) values back to floating-point numbers.
    
    Parameters:
      arr: numpy array of type int16.
      frac_bits: number of fractional bits (default is 15 for Q15 format).
      
    Returns:
      numpy array of floats.
    """
    return arr.astype(np.float32) / MAX_INT


def fixed_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Add two Q15 fixed-point numbers represented as int16 arrays.
    """
    # Convert to int32 to prevent overflow during addition
    result = a.astype(np.int32) + b.astype(np.int32)
    # Clip to 16-bit range
    # result = np.clip(result, MIN_INT, MAX_INT).astype(np.int16)
    return np.round(result).astype(np.int16)



def fixed_mul(a: np.ndarray, b: np.ndarray, frac_bits: int = fraction) -> np.ndarray:
    """
    Multiply two Q15 fixed-point numbers represented as int16 arrays.
    
    The product of two Q15 numbers is a Q30 number, so we shift right by `frac_bits`
    (with rounding) to return to Q15.
    """
    # Multiply as int32 to prevent overflow
    product = a.astype(np.int32) * b.astype(np.int32)
    # Add rounding offset (1 << (frac_bits - 1)) for round-to-nearest
    # product = (product + (1 << (frac_bits - 1))) >> frac_bits
    # Clip to 16-bit range and convert back to int16
    # product = np.clip(np.round(product), MIN_INT, MAX_INT).astype(np.int16)
    return np.round(product).astype(np.int16)


def fixed_dot(a: np.ndarray, b: np.ndarray, frac_bits: int = fraction) -> np.int16:
    """
    Compute the dot product of two Q15 fixed-point vectors.
    
    Parameters:
      a, b: numpy arrays of type int16 representing Q15 fixed-point numbers.
      frac_bits: number of fractional bits (default 15 for Q15).
      
    Returns:
      The dot product result in Q15 fixed-point format (int16).
    
    Process:
      1. Convert a and b to int32 and compute np.dot, resulting in a Q30 value.
      2. Apply rounding: if the accumulator is non-negative, add a positive offset;
         if negative, subtract the offset.
      3. Shift right by frac_bits to convert from Q30 back to Q15.
      4. Clip the result to the valid 16-bit range.
    """
    # Compute dot product with an accumulator in int32.
    acc = np.dot(a.astype(np.int32), b.astype(np.int32))
    
    # Apply rounding in a sign-sensitive way.
    # For non-negative sums: add offset before shifting.
    # For negative sums: subtract offset before shifting.
    if acc >= 0:
        result = (acc + (1 << (frac_bits - 1))) >> frac_bits
    else:
        result = (acc - (1 << (frac_bits - 1))) >> frac_bits

    # Clip result to int16 range and convert to int16.
    result = np.round(acc).astype(np.int16) #np.clip(np.round(acc), MIN_INT, MAX_INT).astype(np.int16)
    return result


def fixed_matvec(matrix: np.ndarray, vector: np.ndarray, frac_bits: int = fraction) -> np.ndarray:
    """
    Perform matrix-vector multiplication in Q15 fixed-point arithmetic.
    
    Parameters:
      matrix: numpy array of shape (m, n) of type int16 representing Q15 numbers.
      vector: numpy array of shape (n,) of type int16 representing Q15 numbers.
      frac_bits: number of fractional bits (default is 15 for Q15).
      
    Returns:
      A numpy array of shape (m,) of type int16 representing the result in Q15 format.
      
    Process:
      1. Convert the matrix and vector to int32 to safely accumulate the products.
      2. Multiply and sum each row (dot product) to produce a Q30 intermediate result.
      3. Apply sign-sensitive rounding by adding (or subtracting) an offset.
      4. Shift right by 15 bits to convert back to Q15.
      5. Clip the results to the valid 16-bit range.
    """
    # Compute the matrix-vector product as int32; each dot product produces a Q30 result.
    acc = matrix.astype(np.int32).dot(vector.astype(np.int32))

    
    # Compute the rounding offset.
    offset = 1 << (frac_bits - 1)
    
    # Apply sign-sensitive rounding:
    # For non-negative values: add offset, then shift.
    # For negative values: subtract offset, then shift.
    acc_rounded = np.where(acc >= 0, acc + offset, acc - offset)
    
    # Shift right by frac_bits (15) to convert Q30 back to Q15.
    result = acc_rounded >> frac_bits
    
    # Clip the results to the valid int16 range and convert type.
    # result = np.clip(result, MIN_INT, MAX_INT).astype(np.int16)
    
    return np.round(acc).astype(np.int16)



def tanh(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, -1*quantization, quantization)


# np.identity(hiddenSize)
# np.zeros((hiddenSize,hiddenSize))
# np.random.randint(min,max,size=(hiddenSize, hiddenSize))
uf = np.eye(hiddenSize, hiddenSize) #np.random.randint(min,max,size=(hiddenSize, hiddenSize))
np.savetxt(fileLoc + 'uf.csv', uf.astype(np.int16), fmt='%i', delimiter=',')

uf2 = np.reshape(uf, (hiddenSize//parDeg, hiddenSize*parDeg))
np.savetxt(fileLoc + 'uf2.csv', uf2.astype(np.int16), fmt='%i', delimiter=',')

ui = np.eye(hiddenSize, hiddenSize) #np.random.randint(min,max,size=(hiddenSize, hiddenSize))
np.savetxt(fileLoc + 'ui.csv', ui.astype(np.int16), fmt='%i', delimiter=',')

ui2 = np.reshape(ui, (hiddenSize//parDeg, hiddenSize*parDeg))
np.savetxt(fileLoc + 'ui2.csv', ui2.astype(np.int16), fmt='%i', delimiter=',')

uo = np.eye(hiddenSize, hiddenSize) #np.random.randint(min,max,size=(hiddenSize, hiddenSize))
np.savetxt(fileLoc + 'uo.csv', uo.astype(np.int16), fmt='%i', delimiter=',')

uo2 = np.reshape(uo, (hiddenSize//parDeg, hiddenSize*parDeg))
np.savetxt(fileLoc + 'uo2.csv', uo2.astype(np.int16), fmt='%i', delimiter=',')

uc = np.eye(hiddenSize, hiddenSize) #np.random.randint(min,max,size=(hiddenSize, hiddenSize))
np.savetxt(fileLoc + 'uc.csv', uc.astype(np.int16), fmt='%i', delimiter=',')

uc2 = np.reshape(uc, (hiddenSize//parDeg, hiddenSize*parDeg))
np.savetxt(fileLoc + 'uc2.csv', uc2.astype(np.int16), fmt='%i', delimiter=',')


# np.zeros((hiddenSize,inputSize))
# np.eye(hiddenSize, inputSize)
# np.random.randint(min,max,size=(hiddenSize, inputSize))

wf = np.eye(hiddenSize, inputSize) #np.random.randint(min,max,size=(hiddenSize, inputSize))
# np.random.randint(min,max,size=(hiddenSize, inputSize), dtype=np.int16)
np.savetxt(fileLoc + 'wf.csv', wf.astype(np.int16), fmt='%i', delimiter=',')

wf2 = np.reshape(wf, (hiddenSize//parDeg, inputSize*parDeg))
np.savetxt(fileLoc + 'wf2.csv', wf2.astype(np.int16), fmt='%i', delimiter=',')

wi = np.eye(hiddenSize, inputSize) #np.random.randint(min,max,size=(hiddenSize, inputSize))
np.savetxt(fileLoc + 'wi.csv', wi.astype(np.int16), fmt='%i', delimiter=',')

wi2 = np.reshape(wi, (hiddenSize//parDeg, inputSize*parDeg))
np.savetxt(fileLoc + 'wi2.csv', wi2.astype(np.int16), fmt='%i', delimiter=',')

wo = np.eye(hiddenSize, inputSize) #np.random.randint(min,max,size=(hiddenSize, inputSize))
np.savetxt(fileLoc + 'wo.csv', wo.astype(np.int16), fmt='%i', delimiter=',')

wo2 = np.reshape(wo, (hiddenSize//parDeg, inputSize*parDeg))
np.savetxt(fileLoc + 'wo2.csv', wo2.astype(np.int16), fmt='%i', delimiter=',')

wc = np.eye(hiddenSize, inputSize) #np.random.randint(min,max,size=(hiddenSize, inputSize))
np.savetxt(fileLoc + 'wc.csv', wc.astype(np.int16), fmt='%i', delimiter=',')

wc2 = np.reshape(wc, (hiddenSize//parDeg, inputSize*parDeg))
np.savetxt(fileLoc + 'wc2.csv', wc2.astype(np.int16), fmt='%i', delimiter=',')


wy = np.eye(outputSize, hiddenSize)
# np.random.randint(min,max,size=(outputSize, hiddenSize))
np.savetxt(fileLoc + 'Wy.csv', wy.astype(np.int16), fmt='%i', delimiter=',')

# bh = np.random.randint(min,max,size=(1,hiddenSize))
# np.savetxt(fileLoc + 'bh.csv', bh.astype(np.int16), fmt='%i', delimiter=',')

# by = np.random.randint(min,max,size=(1,outputSize))
# np.savetxt(fileLoc + 'by.csv', by.astype(np.int16), fmt='%i', delimiter=',')

h0 = np.zeros((hiddenSize,1), dtype=np.int16) #np.random.randint(-1,2,size=(hiddenSize, 1))
# np.zeros((hiddenSize,1), dtype=np.int16)
# np.array([[(i)%2==0] for i in range(hiddenSize)])
c0 = np.zeros((hiddenSize,1), dtype=np.int16)
# np.zeros((hiddenSize,1), dtype=np.int16)
# np.zeros((hiddenSize,1), dtype=np.int16)

# np.zeros((hiddenSize,1), dtype=np.int16) 
# np.random.randint(min,max,size=(1,hiddenSize))

np.savetxt(fileLoc + 'state.csv', np.concatenate([h0,c0]).T.astype(np.int16), fmt='%i', delimiter=',')
np.savetxt(fileLoc + 'stateOut.csv', np.concatenate([h0]).astype(np.int16), fmt='%i', delimiter=',')


x =  -1*np.ones((length,inputSize), dtype=np.int16) #np.random.randint(0,2,size=(length,inputSize))
# nnp.eye(length,inputSize, dtype=np.int16) 
# np.ones((length,inputSize), dtype=np.int16) 
# np.array([[j for i in range(inputSize)] for j in range(length)])

# np.array([[(i+j)%2 for i in range(inputSize)] for j in range(length)])

# Lower triangle
# np.tril(np.random.randint(min,max,size=(length,inputSize)), 0)

# Upper triangle
# np.triu(np.random.randint(min,max,size=(length,inputSize)), 0)

np.savetxt(fileLoc + 'x.csv', x.astype(np.int16), fmt='%i', delimiter=',')

with open(fileLoc + 'x2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the array multiple times
    for _ in range(seqLength):
        writer.writerows(x.astype(np.int16))



h = h0.reshape(hiddenSize,) #np.transpose(h0)
c = c0.reshape(hiddenSize,) #np.transpose(h0)
y = []

print("uf", uf)
print("wf", wf)


for in_x in x:
    # print("h", h.astype(np.int16))
    # print("x", in_x)

    f = (uf.dot(h)+ wf.dot(in_x))
    # print("F", f)

    i = (ui.dot(h)+ wi.dot(in_x))

    o = (uo.dot(h)+ wo.dot(in_x))

    c_prime = tanh(uc.dot(h)+ wc.dot(in_x))

    c = f*c+i*c_prime
    print("c,", c.astype(np.int16))

    h = tanh(c*o)
    print("h,",h.astype(np.int16))

    y += [np.concatenate([h,c])]



Y = np.array(y)



# y = np.random.randint(min,max,size=(length,outputSize))
np.savetxt(fileLoc + 'y.csv', Y.astype(np.int16), fmt='%i', delimiter=',')

with open(fileLoc + 'y2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the array multiple times
    for _ in range(seqLength):
        writer.writerows(Y.astype(np.int16))
