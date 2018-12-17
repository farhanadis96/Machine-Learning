# a=[1,2,3,4,5,6,7,8,9,10]
# print("My name is Farhan")

import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import operator
from numpy import float64
from random import shuffle

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
#print(trainData[2])
trainLabel=read_data("train-labels.idx1-ubyte")
# print(trainLabel[3])

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



def find_neighbors(train_shuffle, test_instance, k,trainLabel):
	distances =[]

	length = len(test_instance)-1
	for x in range(int(len(train_shuffle))):
		dist = euc_dist(test_instance, train_shuffle[x],length)
		# print(dist)


		# dist = euc_dist(test_instance, train_shuffle[x], length)
		distances.append((train_shuffle[x],trainLabel[x], dist))
	distances.sort(key=operator.itemgetter(2))
	# print(distances)
	neighbors =[]

	for x in range(k):
			
			neighbors.append(distances[x][1])
	neighbors=np.array(neighbors)
	neighbors=np.reshape(neighbors,(k,1))
	#print(neighbors)
	return neighbors


def sort_Knearest(neighbor_count):
	return sorted(neighbor_count.items(), key=operator.itemgetter(1), reverse=True)


def majority(neighbors):


	neighbor_count ={}



	for m in range(len(neighbors)):


		value =neighbors[m,:]
		#print(value)

		if value in tuple(neighbor_count):
			neighbor_count[tuple(value)] += 1
			
		else:

			neighbor_count[tuple(value)] = 1
			#print(neighbor_count)

		total_neighbors = sort_Knearest(neighbor_count)
	# print(total_neighbors)
	
	return total_neighbors[0][0]







def knn_class(trainData,testData,trainLabel,testLabel):

	k=[1,2,3,4,5,6,7,8,9,10] #run for 10 values of k
	
	for i in k:
		val_correct = 0
		# y_pred=[]
		# y_actu=[]
		for x in range(int(len(testData))):
			neighbors =np.array(find_neighbors(trainData,testData[x], k[i-1],trainLabel))

			pred_label=majority(neighbors)
			# y_pred=pred_label
			# y_actu=testLabel[x]
			if pred_label==testLabel[x]:
				val_correct +=1


		accuracy=val_correct/(int(len(testData)))
		print(accuracy*100)
		


a=[i for i in range(0,60000)]

shuffle(a)
#print(len(a))


train_shuff=np.array(trainData[a])
label_shuff=np.array(trainLabel[a])

# print(train_shuff.shape)
# print("label shape",label_shuff.shape)



for i in range(10):
	testset_cv =train_shuff[(i*6000):(6000 * (i + 1)), :]
	#print(testset_cv.shape)
	label_shuff=np.reshape(label_shuff,(60000,1))
	testset_cvlabel =label_shuff[(i*6000):(6000*(i + 1)), :]
	# print(testset_cvlabel.shape)
	# print(type(testset_cv))
	trainset_cv=[]
	trainset_cv = np.empty((1200,784))
	trainset_cvlabel = np.empty((1200,1))


	for j in range(10):
		if(j!=i):
			part_append=train_shuff[(j * 6000):(6000*(j+1)),:]

			part_labelappend=label_shuff[(j * 6000):(6000*(j+1)),:]

			trainset_cv = np.append(trainset_cv,part_append, 0)

			trainset_cvlabel=np.append(trainset_cvlabel,part_labelappend, 0)
	trainset_cv = trainset_cv[1200:12000,:]
	trainset_cvlabel = trainset_cvlabel[1200:12000, :]
	# print(trainset_cv.shape)
	# print(trainset_cvlabel.shape)
	knn_class(trainset_cv,testset_cv,trainset_cvlabel,testset_cvlabel)







