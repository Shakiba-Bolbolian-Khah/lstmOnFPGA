
import numpy as np
import csv

temp = 12
hiddenSize = temp
inputSize = temp
outputSize = temp
length = int(temp/2)
seqLength = temp

parDeg1 = 2
parDeg2 = 4
parDeg3 = 2
min = 0
max = 2
fileLoc = 'data/'

wh = np.identity(hiddenSize)
# np.identity(hiddenSize)
# np.zeros((hiddenSize,hiddenSize))

# np.random.randint(min,max,size=(hiddenSize, hiddenSize))
np.savetxt(fileLoc + 'Wh.csv', wh.astype(int), fmt='%i', delimiter=',')

wh2 = np.reshape(wh, (hiddenSize//parDeg1, hiddenSize*parDeg1))
np.savetxt(fileLoc + 'Wh2.csv', wh2.astype(int), fmt='%i', delimiter=',')

wh4 = np.reshape(wh, (hiddenSize//parDeg2, hiddenSize*parDeg2))
np.savetxt(fileLoc + 'Wh4.csv', wh4.astype(int), fmt='%i', delimiter=',')

wh8 = np.reshape(wh, (hiddenSize//parDeg3, hiddenSize*parDeg3))
np.savetxt(fileLoc + 'Wh8.csv', wh8.astype(int), fmt='%i', delimiter=',')

wx = np.eye(hiddenSize, inputSize)
# np.zeros((hiddenSize,inputSize))
# np.eye(hiddenSize, inputSize)

# np.random.randint(min,max,size=(hiddenSize, inputSize))
np.savetxt(fileLoc + 'Wx.csv', wx.astype(int), fmt='%i', delimiter=',')

wx2 = np.reshape(wx, (hiddenSize//parDeg1, inputSize*parDeg1))
np.savetxt(fileLoc + 'Wx2.csv', wx2.astype(int), fmt='%i', delimiter=',')

wx4 = np.reshape(wx, (hiddenSize//parDeg2, inputSize*parDeg2))
np.savetxt(fileLoc + 'Wx4.csv', wx4.astype(int), fmt='%i', delimiter=',')

wx8 = np.reshape(wx, (hiddenSize//parDeg3, inputSize*parDeg3))
np.savetxt(fileLoc + 'Wx8.csv', wx8.astype(int), fmt='%i', delimiter=',')

wy = np.eye(outputSize, hiddenSize)
# np.random.randint(min,max,size=(outputSize, hiddenSize))
np.savetxt(fileLoc + 'Wy.csv', wy.astype(int), fmt='%i', delimiter=',')

wy2 = np.reshape(wy, (outputSize//parDeg1, hiddenSize*parDeg1))
np.savetxt(fileLoc + 'Wy2.csv', wy2.astype(int), fmt='%i', delimiter=',')

wy4 = np.reshape(wy, (outputSize//parDeg2, hiddenSize*parDeg2))
np.savetxt(fileLoc + 'Wy4.csv', wy4.astype(int), fmt='%i', delimiter=',')

wy8 = np.reshape(wy, (outputSize//parDeg3, hiddenSize*parDeg3))
np.savetxt(fileLoc + 'Wy8.csv', wy8.astype(int), fmt='%i', delimiter=',')


# bh = np.random.randint(min,max,size=(1,hiddenSize))
# np.savetxt(fileLoc + 'bh.csv', bh.astype(int), fmt='%i', delimiter=',')

# by = np.random.randint(min,max,size=(1,outputSize))
# np.savetxt(fileLoc + 'by.csv', by.astype(int), fmt='%i', delimiter=',')

h0 = np.ones((hiddenSize,1), dtype=int)
# np.zeros((hiddenSize,1), dtype=int) 
# np.ones((hiddenSize,1), dtype=int)
# np.random.randint(min,max,size=(1,hiddenSize))

np.savetxt(fileLoc + 'h0.csv', h0.T.astype(int), fmt='%i', delimiter=',')
np.savetxt(fileLoc + 'h0_out.csv', h0.astype(int), fmt='%i', delimiter=',')

# x_temp = np.triu(np.random.randint(min,max,size=(length,5)), 0)
# x = np.pad(x_temp, ((0,0),(inputSize-5,0)))

x = np.ones((length,inputSize), dtype=int) 

# np.array([[(i+j)%2 if j%5==0 else 0 for i in range(inputSize)] for j in range(length)])
# np.random.randint(min,max,size=(length,inputSize))
# np.array([[(i+j)%2 for i in range(inputSize)] for j in range(length)])
# np.ones((length,inputSize), dtype=int) 

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



h = h0.reshape(hiddenSize,) #np.transpose(h0)
y = []

for in_x in x:
    # h = h + in_x
    h = wh.dot(h)+ wx.dot(np.transpose(in_x))
    # y += [h]
    y += [wy.dot(h)]


Y = np.array(y)



# y = np.random.randint(min,max,size=(length,outputSize))
np.savetxt(fileLoc + 'y.csv', Y.astype(int), fmt='%i', delimiter=',')

with open(fileLoc + 'y2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the array multiple times
    for _ in range(seqLength):
        writer.writerows(Y.astype(int))



with open(fileLoc + 'x.csv', 'r') as infile:
    reader = csv.reader(infile)
    data = list(reader)  # Read all the data into a list
