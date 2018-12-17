import struct
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import operator
from numpy import float64
from random import shuffle
import pandas as pd

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

original_train=np.array(original_train,dtype=float64)
original_test=np.array(original_test,dtype=float64)

# def euc_dist(inst1, inst2):
#      distance = np.linalg.norm(np.array(inst1)-np.array(inst2))
#      return math.sqrt(distance)





def euclideanDistance(inst1, inst2, length):
	distance = 0
	#print(inst1.shape)
	for x in range(length):
		distance += pow((inst1[1,x] - inst2[1,x]), 2)
	return math.sqrt(distance)



def find_neighbors(trainingset, testinstance, k,trainLabel):
	distances_concat =[]

	length = len(testinstance)-1
	for x in range(int(len(trainingset))):

		sample = np.array(trainingset[x, :])
		# print(sample.shape)
		sample = np.pad(sample, pad_width=1, mode='constant', constant_values=1)

		# print(sample.shape)

		i1 = np.array(sample[0:28, 0:28])
		i2 = np.array(sample[0:28, 1:29])
		i3 = np.array(sample[0:28, 2:30])
		i4 = np.array(sample[1:29, 0:28])
		i5 = np.array(sample[1:29, 1:29])
		i6 = np.array(sample[1:29, 2:30])
		i7 = np.array(sample[2:30, 0:28])
		i8 = np.array(sample[2:30, 1:29])
		i9 = np.array(sample[2:30, 2:30])


		distances = np.empty([1, 1])
		image = np.array([i1, i2, i3, i4, i5, i6, i7, i8, i9])
		# print(image.shape)
		for i in range(9):
			#test_part = np.reshape(image[i], (1, 784))
			#testinstance=np.reshape(testinstance,(1,784))
			dist =euclideanDistance(testinstance,image[i],length)
			dist = np.reshape(dist, (1, 1))
			distances = np.append(distances, dist, 0)
		# print(distances.shape)
		distances = np.array(distances[1:10, :])
		# print(distances)
		min_dist = np.amin(distances)
		# print(min_dist)
		# print(dist)


		# dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances_concat.append((trainingset[x],trainLabel[x],min_dist))
	distances_concat.sort(key=operator.itemgetter(2))
	neighbors =[]
	# print(distances)


	for x in range(k):
			# neighbors.append(distances[x][0])
			neighbors.append(distances_concat[x][1])
	neighbors=np.array(neighbors)
	neighbors=np.reshape(neighbors,(4,1))
	# print(neighbors)
	return neighbors


def sort_Knearest(neighbor_count):
	return sorted(neighbor_count.items(), key=operator.itemgetter(1), reverse=True)


def majority(neighbors):


	neighbor_count = {}



	for m in range(len(neighbors)):



		value =neighbors[m,:]


		if value in tuple(neighbor_count):
			neighbor_count[tuple(value)] += 1
		else:
			neighbor_count[tuple(value)] = 1


		total = sort_Knearest(neighbor_count)
	# print(total)
	#print(total[0][0])
	return total[0][0]


def plt_confmat(confusion_matrix, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(confusion_matrix, cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(confusion_matrix.columns))
    plt.xticks(tick_marks, confusion_matrix.columns, rotation=45)
    plt.yticks(tick_marks, confusion_matrix.index)
    plt.ylabel(confusion_matrix.index.name)
    plt.xlabel(confusion_matrix.columns.name)
    plt.show()


k=4


correct_v = 0
y_pred=[]
y_actu=[]

for x in range(int(len(original_test))):
	neighbors =np.array(find_neighbors(original_train,original_test[x], k,trainLabel))

	label_pred=majority(neighbors)
	y_pred.append(label_pred)
	y_actu.append(testLabel[x])
	if label_pred==testLabel[x]:
		correct_v +=1
	print("image %d is done"%(x))



accuracy=correct_v/(int(len(original_test)))
print(accuracy*100)
x1=pd.Series(y_actu, name='Actual')
y1=pd.Series(y_pred, name='Predicted')
confusion_matrix = pd.crosstab(x1,y1,margins=True)
print(confusion_matrix)
plt_confmat(confusion_matrix)





