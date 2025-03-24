import numpy as np

IO_ROOF = 6.9
TOTAL_DSP = 1518



# Number of multiplications in one LSTM layer, does not include the dense layer
def num_ops_lstm(input_size, hidden_size, hidden_units):
    cell_ops = 4 * hidden_size * ( input_size + hidden_size) + 3 * hidden_size
    layer_ops = hidden_units * cell_ops

    return layer_ops


# Only the memory used to read weight matrices and load them into FPGA. Writing back is not considered
def mem_lstm(input_size, hidden_size, hidden_units, bit_width):

    # input weights and hidden weights and bias vectors
    mem_weights = 4 * (hidden_size * hidden_size + input_size * hidden_size + hidden_size)
    
    return mem_weights * bit_width / 8


#Number of multiplication in one dense layer. If it is applied to all outputs of lstm, is_all_output is set to True
def num_ops_dense(input_size, output_size, hidden_units, is_all_output: bool = True):
    
    cell_ops = input_size * output_size
    layer_ops = cell_ops * hidden_units if is_all_output else cell_ops

    return layer_ops


# Memory read by dense layer
def mem_dense(input_size, output_size, bit_width):
    
    return (input_size * output_size + output_size) * bit_width / 8


# If the model is many-to-one, the flag is set to false
def writing_mem(output_size, hidden_units, bit_width, is_all_output: bool = True):

    mem_output = output_size * hidden_units if is_all_output else output_size
    return mem_output * bit_width / 8


def reading_mem(input_size, hidden_units, bit_width):

    mem_input = input_size * hidden_units
    return mem_input * bit_width / 8


def cal_ops_mem(input_size, hidden_size, hidden_units, output_size, num_tests, bit_width, is_all_output):

    model_op = num_ops_lstm(input_size, hidden_size, hidden_units)
    model_op += num_ops_dense(input_size, output_size, hidden_units, is_all_output)

    total_ops = num_tests * model_op

    lstm_mem = mem_lstm(input_size, hidden_size, hidden_units, bit_width)
    dense_mem = mem_dense (input_size, output_size, bit_width)
    input_mem = num_tests * reading_mem(input_size, hidden_units, bit_width)
    output_mem = num_tests * writing_mem(output_size, hidden_units, bit_width, is_all_output)

    total_mem = lstm_mem + dense_mem + input_mem + output_mem

    return total_ops, total_mem


# Here's the details of the performance metric
# Execution time = clock_cycle / frequency
# GOPS = Total operations / Execution time
# OPC = Total operations / clock_cycle
# OI (Operation Intensity) = Total opertion / Total mem

def compute_performance_metrics(model_name, op_nums, memory, frequency, clock_cycle, dsp_num, mul_per_dsp):
    
    perf = {}
    perf['model'] = model_name
    perf['GOPS'] = round((10**-9) * op_nums * frequency / clock_cycle, 3)
    perf['OPC'] = round(op_nums / clock_cycle, 3)

    perf['OI'] = round(op_nums / memory, 3)
    perf['DSP_EFF'] = round(100 *op_nums / (dsp_num * mul_per_dsp * clock_cycle), 3)

    perf['latency'] = round(clock_cycle / frequency, 5)

    return perf


def mnist_model():
    input_size = 28
    hidden_size = 32
    hidden_units = 28
    bit_width = 16
    output_size = 10
    is_all_output = False
    num_tests = 10000

    mnist_ops, mnist_mems = cal_ops_mem(input_size, hidden_size, hidden_units, output_size, num_tests, bit_width, is_all_output)

    print("==================== Analyaing performance for MNIST dataset ====================")
    print("* Number of Operations:", mnist_ops)
    print("\n")
    #For SHIR model:
    clock_cycle = 23480000 #15510921
    frequency = 200 * (10**6)
    dsp_num = 165
    mul_per_dsp = 2
    model_name = "SHIR-MNIST"
    perf_shir = compute_performance_metrics(model_name, mnist_ops, mnist_mems, frequency, clock_cycle, dsp_num, mul_per_dsp)
    print(perf_shir)

    #For SHIR model:
    clock_cycle = 21630000
    frequency = 157 * (10**6)
    dsp_num = 195
    mul_per_dsp = 2
    model_name = "HLS4ML-MNIST"
    perf_hls4ml = compute_performance_metrics(model_name, mnist_ops, mnist_mems, frequency, clock_cycle, dsp_num, mul_per_dsp)
    print(perf_hls4ml)
    print("\n=================================================================================")

mnist_model()

def imdb_model():
    input_size = 100
    hidden_size = 128
    hidden_units = 200
    bit_width = 16
    output_size = 1
    is_all_output = False
    num_tests = 1000

    imdb_ops, imdb_mems = cal_ops_mem(input_size, hidden_size, hidden_units, output_size, num_tests, bit_width, is_all_output)

    print("==================== Analyaing performance for IMDB dataset ====================")
    print("* Number of Operations:", imdb_ops)
    print("\n")
    #For SHIR model:
    clock_cycle = 30585000 #15510921
    frequency = 200 * (10**6)
    dsp_num = 601
    mul_per_dsp = 2
    model_name = "SHIR-IMDB"
    perf_shir = compute_performance_metrics(model_name, imdb_ops, imdb_mems, frequency, clock_cycle, dsp_num, mul_per_dsp)
    print(perf_shir)

    #For SHIR model:
    clock_cycle = 21630000
    frequency = 157 * (10**6)
    dsp_num = 195
    mul_per_dsp = 2
    model_name = "HLS4ML-IMDB"
    perf_hls4ml = compute_performance_metrics(model_name, imdb_ops, imdb_mems, frequency, clock_cycle, dsp_num, mul_per_dsp)
    print(perf_hls4ml)
    print("\n=================================================================================")

imdb_model()