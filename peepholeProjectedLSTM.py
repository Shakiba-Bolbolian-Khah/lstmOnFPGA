
import numpy as np
import csv

temp = 8

hiddenSize = 8
inputSize = 8
projectedSize = 8
length = int(temp/2)
seqLength = temp

parDeg = 4
min = 0
max = 2
fileLoc = 'data/'


# ----- Weight Matrices for hidden state h -----

# np.identity(hiddenSize)
# np.zeros((hiddenSize,hiddenSize))
# np.random.randint(min,max,size=(hiddenSize, hiddenSize))
uf = np.zeros((hiddenSize,projectedSize), dtype=int)
np.savetxt(fileLoc + 'uf0.csv', uf.astype(int), fmt='%i', delimiter=',')

uf2 = np.reshape(uf, (hiddenSize//parDeg, projectedSize*parDeg))
np.savetxt(fileLoc + 'uf1.csv', uf2.astype(int), fmt='%i', delimiter=',')

ui = np.zeros((hiddenSize,projectedSize), dtype=int)
np.savetxt(fileLoc + 'ui0.csv', ui.astype(int), fmt='%i', delimiter=',')

ui2 = np.reshape(ui, (hiddenSize//parDeg, projectedSize*parDeg))
np.savetxt(fileLoc + 'ui1.csv', ui2.astype(int), fmt='%i', delimiter=',')

uo = np.zeros((hiddenSize,projectedSize), dtype=int)
np.savetxt(fileLoc + 'uo0.csv', uo.astype(int), fmt='%i', delimiter=',')

uo2 = np.reshape(uo, (hiddenSize//parDeg, projectedSize*parDeg))
np.savetxt(fileLoc + 'uo1.csv', uo2.astype(int), fmt='%i', delimiter=',')

uc = np.zeros((hiddenSize,projectedSize), dtype=int)
np.savetxt(fileLoc + 'uc0.csv', uc.astype(int), fmt='%i', delimiter=',')

uc2 = np.reshape(uc, (hiddenSize//parDeg, projectedSize*parDeg))
np.savetxt(fileLoc + 'uc1.csv', uc2.astype(int), fmt='%i', delimiter=',')


# np.zeros((hiddenSize,inputSize))
# np.eye(hiddenSize, inputSize)
# np.random.randint(min,max,size=(hiddenSize, inputSize))

# ----- Weight matrices for input X -----

wf = np.eye(hiddenSize, inputSize, dtype=int)
np.savetxt(fileLoc + 'wf0.csv', wf.astype(int), fmt='%i', delimiter=',')

wf2 = np.reshape(wf, (hiddenSize//parDeg, inputSize*parDeg))
np.savetxt(fileLoc + 'wf1.csv', wf2.astype(int), fmt='%i', delimiter=',')

wi = np.eye(hiddenSize, inputSize, dtype=int)
np.savetxt(fileLoc + 'wi0.csv', wi.astype(int), fmt='%i', delimiter=',')

wi2 = np.reshape(wi, (hiddenSize//parDeg, inputSize*parDeg))
np.savetxt(fileLoc + 'wi1.csv', wi2.astype(int), fmt='%i', delimiter=',')

wo = np.eye(hiddenSize, inputSize, dtype=int)
np.savetxt(fileLoc + 'wo0.csv', wo.astype(int), fmt='%i', delimiter=',')

wo2 = np.reshape(wo, (hiddenSize//parDeg, inputSize*parDeg))
np.savetxt(fileLoc + 'wo1.csv', wo2.astype(int), fmt='%i', delimiter=',')

wc = np.eye(hiddenSize, inputSize, dtype=int)
np.savetxt(fileLoc + 'wc0.csv', wc.astype(int), fmt='%i', delimiter=',')

wc2 = np.reshape(wc, (hiddenSize//parDeg, inputSize*parDeg))
np.savetxt(fileLoc + 'wc1.csv', wc2.astype(int), fmt='%i', delimiter=',')

# ----- Weight matrices for peephole connection on c vectors -----

cf = np.zeros((hiddenSize,1), dtype=int)
# np.eye(hiddenSize,1, dtype=int)
# np.random.randint(min,max,size=(hiddenSize, 1))
np.savetxt(fileLoc + 'cf.csv', cf.reshape(1,hiddenSize).astype(int), fmt='%i', delimiter=',')

ci =  np.zeros((hiddenSize,1), dtype=int)
# np.random.randint(min,max,size=(hiddenSize, 1))
np.savetxt(fileLoc + 'ci.csv', ci.reshape(1,hiddenSize).astype(int), fmt='%i', delimiter=',')

co =  np.zeros((hiddenSize,1), dtype=int)
# np.random.randint(min,max,size=(hiddenSize, 1))
np.savetxt(fileLoc + 'co.csv', co.reshape(1,hiddenSize).astype(int), fmt='%i', delimiter=',')

# ----- Weight matrices for projection -----


wy = np.eye(projectedSize, hiddenSize)
# np.random.randint(min,max,size=(projectedSize, hiddenSize))
np.savetxt(fileLoc + 'wy.csv', wy.astype(int), fmt='%i', delimiter=',')

# bh = np.random.randint(min,max,size=(1,hiddenSize))
# np.savetxt(fileLoc + 'bh.csv', bh.astype(int), fmt='%i', delimiter=',')

# by = np.random.randint(min,max,size=(1,projectedSize))
# np.savetxt(fileLoc + 'by.csv', by.astype(int), fmt='%i', delimiter=',')

h0 = np.zeros((projectedSize,1), dtype=int)
# np.zeros((hiddenSize,1), dtype=int)
# np.array([[(i)%2==0] for i in range(hiddenSize)])
c0 = np.zeros((hiddenSize,1), dtype=int)
# np.zeros((hiddenSize,1), dtype=int)
# np.zeros((hiddenSize,1), dtype=int)

# np.zeros((hiddenSize,1), dtype=int) 
# np.random.randint(min,max,size=(1,hiddenSize))

np.savetxt(fileLoc + 'state.csv', np.concatenate([h0,c0]).T.astype(int), fmt='%i', delimiter=',')
np.savetxt(fileLoc + 'stateOut.csv', np.concatenate([h0]).astype(int), fmt='%i', delimiter=',')


x =  np.eye(length,inputSize, dtype=int) 
# np.random.randint(0,2,size=(length,inputSize))
# np.eye(length,inputSize, dtype=int) 
# np.ones((length,inputSize), dtype=int) 
# np.array([[j for i in range(inputSize)] for j in range(length)])

# np.array([[(i+j)%2 for i in range(inputSize)] for j in range(length)])

# Lower triangle
# np.tril(np.random.randint(min,max,size=(length,inputSize)), 0)

# Upper triangle
# np.triu(np.random.randint(min,max,size=(length,inputSize)), 0)

np.savetxt(fileLoc + 'x.csv', x.astype(int), fmt='%i', delimiter=',')

with open(fileLoc + 'x2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the array multiple times
    for _ in range(seqLength):
        writer.writerows(x.astype(int))



h = h0.reshape(projectedSize,) #np.transpose(h0)
c = c0.reshape(hiddenSize,) #np.transpose(h0)
y = []

for in_x in x:

    f = uf.dot(h)+ wf.dot(np.transpose(in_x)) + (cf.reshape(hiddenSize,)*c)
    # print(cf.reshape(hiddenSize,), c, cf.reshape(hiddenSize,)*c)
    # print("f: ", f)

    i = ui.dot(h)+ wi.dot(np.transpose(in_x)) + (ci.reshape(hiddenSize,)*c)
    # print("i: ", i)

    o = uo.dot(h)+ wo.dot(np.transpose(in_x)) + (co.reshape(hiddenSize,)*c)
    # print("o: ", o)

    c_prime = uc.dot(h)+ wc.dot(np.transpose(in_x))
    # print("c_prime: ", c_prime)

    c = (f*c) + (i*c_prime)
    # print("c: ", c)

    h = wy.dot(c*o)
    # print("h: ", h)

    y += [np.concatenate([h,c])]
    print("Y:  ", y)



Y = np.array(y)



# y = np.random.randint(min,max,size=(length,projectedSize))
np.savetxt(fileLoc + 'y.csv', Y.astype(int), fmt='%i', delimiter=',')

with open(fileLoc + 'y2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the array multiple times
    for _ in range(seqLength):
        writer.writerows(Y.astype(int))
