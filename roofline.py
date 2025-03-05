# Addition of 2 MVM: W_x*X .+ W_h*H
# Num of operations per MVM for M*N matrice: N(2M-1) (N is the final dimension)
# Num of operations for this benchmark: STM_LEN [ 2N(M+N-1) + N ]
# Num of read/written data: STM_LEN (M + N)


m = 512
# int(input("m (size of hidden)?\n"))
n = 512
# int(input("n (size of input)?\n"))
p = 512
# int(input("p (size of output)?\n"))
hiddenUnits = 256
# int(input("stream length (number of elements in the stream)?\n"))
cc = 214642 #58266 #42602
# int(input("Number of CC?\n"))
bitWidth = 16
# int(input("Bitwidth of data?\n"))
dsp = 192
# int(input("Number of used DSPs:\n"))
mulPerDSP = 2
# float(input("Multiplications per DSP:\n"))


ioRoof = 6.9
totalDSP = 1518

ops_h = m*(n+m) #m*( 2*n+ 2*m- 1)
ops_y = p*m # p*( 2*m -1)
ops = hiddenUnits *( ops_h+ ops_y)
print("#OPS:", ops)
print("#OPS/cycle:", ops/cc)

f = 200 * (10**6)

gops = ops * f / cc
print("***Performance***\n#OPS/s:", gops, "\n#GOPS/s:", gops * 10**(-9))


mem = ((hiddenUnits * n)+(m*m)+(n*m)+(p*m)) * bitWidth / 8

oi = ops/ mem
print("***Operation (Computational) Intensity***\n\#OI:", oi)


muls = hiddenUnits*( m*( m+ n+ p))
# mulPerDSP = 1 if bitWidth > 16 else int(36/bitWidth)
print("Mul/DSP:", mulPerDSP)
print("Mul: ", muls)

dspEff = muls / (dsp * mulPerDSP*cc)
print("DSP efficiency:", dspEff)


peakPerf = totalDSP * mulPerDSP * f * (10 ** (-9))
print("(Computational Roof) Peak Performance: #GOPs/s", peakPerf)


print("(I/O Bandwidth Roof): (GB/sec)", ioRoof)

rooflinePoint = min(peakPerf, oi * ioRoof)
print("***Roofline Point***\nX: ", oi, ", Y:  ", gops * 10**(-9), ",  Roofline Point:  ", rooflinePoint)
