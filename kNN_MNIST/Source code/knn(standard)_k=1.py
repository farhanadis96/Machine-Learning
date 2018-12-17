import struct
import numpy as np
import pandas as pd
from numpy import float64
import matplotlib.pyplot as plt
import random
import math
import operator

'''Function to read MNIST dataset files'''

def read_data(filename):
    with open(filename, 'rb') as f:
        zero,data_type,dims=struct.unpack('>HBB',f.read(4))
        shape=tuple(struct.unpack('>I',f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(),dtype=np.uint8).reshape(shape)

original_train=read_data("train-images.idx3-ubyte")
#print(original_train.shape)
#print(original_train)
trainData=np.reshape(original_train,(60000,28*28))
#print(trainData[10])
trainLabel=read_data("train-labels.idx1-ubyte")
# print(trainLabel[10])

original_test=read_data("t10k-images.idx3-ubyte")
#print(original_test.shape)
testData=np.reshape(original_test,(10000,28*28))


#print(len(testData))
testLabel=read_data("t10k-labels.idx1-ubyte")
#print(testLabel[0:10])
# print(original_train[0])
#print(trainData[0])

trainData=np.array(trainData,dtype=float64)
testData=np.array(testData,dtype=float64)






def euc_dist(inst1, inst2, length):
 	distance = 0
 	for x in range(length):
 		distance += pow((inst1[x] - inst2[x]), 2)
 	return math.sqrt(distance)



def find_neighbors(trainData, test_instance, k):
	distances = []

	length = len(test_instance)-1
	for x in range(int(len(trainData))):
		dist = euc_dist(test_instance, trainData[x],length)
		# print(dist)


		
		distances.append((trainData[x],trainLabel[x], dist))
	distances.sort(key=operator.itemgetter(2))
	# print(distances)
	neighbors = []

	for x in range(k):
			
			neighbors.append(distances[x][1])
	# print(neighbors)
	return neighbors


def sort_Knearest(neighbor_count):
	return sorted(neighbor_count.items(), key=operator.itemgetter(1), reverse=True)


def majority(neighbors):


	neighbor_count = {}


	for m in range(len(neighbors)):
		value =neighbors[m]
		if value in neighbor_count:
			neighbor_count[value] += 1
		else:
			neighbor_count[value] = 1

		total_neighbors = sort_Knearest(neighbor_count)
	# print(total_neighbors)
	#print(total_neighbors[0][0])
	return total_neighbors[0][0]







k=1
val_correct=0


for x in range(int(len(testData))):
		neighbors = find_neighbors(trainData,testData[x], k)
		pred_label=majority(neighbors)
		if pred_label==testLabel[x]:
			val_correct +=1


accuracy=val_correct/(int(len(testData)))
print ("Accuracy for k=1 :%2f"%(accuracy*100))







