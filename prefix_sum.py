import random

def writeSeq(seq):
	out = "Seq("
	for i in seq:
		out += str(i) + ","
	out = out[:-1] + ")"
	return out
	
def writeSequence(mat):
	out = "Seq("
	for j in mat:
		out += writeSeq(j) + ","
	out = out[:-1] + ")"
	return out
	
	
def generateSeq(length, inner,  min, max):
	return [random.choices(range(min, max), k=inner) for i in range(length)]
	
from operator import add, mul
from functools import reduce


def prefixSum(x, inner):
	output = []
	accum = [1]*inner
	for j in range(len(x)):
		accum = list( map(mul, accum, x[j]) )
		#accum =   list(map(mul ,accum ,[reduce(lambda z, y: z + y, x[j])]))
		output += [accum]
	return output
	
x = generateSeq(10, 16, 1, 4)
print(type(x[0]))
out = prefixSum(x, 16)

print(writeSequence(x))
print(writeSequence(out))
