
import random

hiddenSize = 256
inputSize = 128
outputSize = 512
hiddenUnit = 1
tanhFrac = 128

def tanh(x):
	return min(max(x, -1*tanhFrac), tanhFrac)

def tanhSeq(seq):
	result = []
	for i in seq:
		result += [tanh(i)]
	return result

def writeSeq(seq):
	out = "Seq("
	for i in seq:
		out += str(i) + ","
	out = out[:-1] + ")"
	return out

def writeMat(mat):
	seq = []
	for j in mat:
		seq += j
	out = writeSeq(seq)
	return out

def writeInput(mat):
	out = "Seq("
	for j in mat:
		out += writeSeq(j) + ","
	out = out[:-1] + ")"
	return out

def generateSeq(length, min, max):
	return [random.choice(range(min, max)) for i in range(length)]

def generateMat(row, col, min, max):
	mat = []
	for i in range(row):
		mat += [[random.choice(range(min, max)) for j in range(col)]]
	return mat

def mvm(mat, vec):
	result = []
	for i in mat:
		r = 0
		for j in range(len(i)):
			r += i[j] * vec[j]
		result += [r]
	return result

def elementWiseAdd(vec1, vec2):
	result = []
	for i in range(len(vec1)):
		result += [vec1[i]+ vec2[i]]
	return result

#Generating h0 from random input
h0 = generateSeq(hiddenSize, 0, 2)
print("h0:  "+ writeSeq(h0))
print("reversed h0:  "+ writeSeq(h0[::-1]))

#Generating bh from random input
bh = generateSeq(hiddenSize, 0, 2)
print("bh:  "+ writeSeq(bh))

#Generating by from random input
by = generateSeq(outputSize, 0, 2)
print("by:  "+ writeSeq(by))

#Generating w_x from random input
wx = generateMat(hiddenSize, inputSize, 0, 3)

rev_wx = rev_wh = [el[::-1] for el in wx[::-1]] 

#print("wx:   ", 	writeMat(wx))
print("reversed wx:\n", writeMat(rev_wx))
print("wx Shape:  "+ str(len(wx)) + ','+ str(len(wx[0])))

#Generating w_h from random input
wh = generateMat(hiddenSize, hiddenSize, 0, 3)
# [[int(i ==j) for j in range(0,hiddenSize)] for i in range(0, hiddenSize)]



rev_wh = [el[::-1] for el in wh[::-1]] 

	
#print("wh:    ", 	writeMat(wh))
print("reversed wh:\n", writeMat(rev_wh))
print("wh Shape:  "+ str(len(wh)) + ','+ str(len(wh[0])))

#Generating w_y from random input
wy = generateMat(outputSize, hiddenSize, 0, 2)

rev_wy = [el[::-1] for el in wy[::-1]] 

#print("wy:    ", writeMat(wy))
print("reversed wy:\n", writeMat(rev_wy))
print("wy Shape:  "+ str(len(wy)) + ','+ str(len(wy[0])))


#Generating x from random input

X = generateMat(hiddenUnit, inputSize, 0, 2)


print("X:\n", writeInput(X))
print("X Shape:  "+ str(len(X)) + ','+ str(len(X[0])))


h = h0
Y = []


for x in X:
	# print("unupdate"+ str(h))
	#h = elementWiseAdd(elementWiseAdd(mvm(wx, x), mvm(wh,h)), bh)
	#y = elementWiseAdd(mvm(wy,h), by)
	

	h = elementWiseAdd(mvm(wx, x), mvm(wh,h))
	y = mvm(wy,h)
	Y += [ tanhSeq(y)]
	# print(h)
	# print(y)
print("output:   \n", writeInput(Y))
print('\n')







		

			
	


