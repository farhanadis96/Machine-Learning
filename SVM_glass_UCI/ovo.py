import pandas as pd
import numpy as np
from sklearn import svm
import random
from numpy import float16
import operator
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
import time
from sklearn.multiclass import OneVsRestClassifier



gdata = pd.read_csv('uci_dataset.csv')
x = gdata.drop(['Type','Id'], axis=1)
y = gdata['Type']
   
np.random.seed(5)
s=np.arange(214)
np.random.shuffle(s)
  
x=x.values
x=np.array(x[s])
    
y=y.values
y=np.array(y[s])


x_train, x_test, y_trainlabel, y_testlabel = train_test_split(x, y, test_size = 0.20,shuffle=False) 
# print(y_test)
# print(y_test.shape)

 



''' RBF'''

Cs = [2**-5,2**-4,2**-3,2**-2,2**-1,1,2**1,2**2,2**3,2**4,2**5,2**6]
gamma_range = [2**-15,2**-13,2**-11,2**-9,2**-7,2**-5,2**-3,2**-1,2**1,2**3,2**5]

b=[]

for C in Cs:
    for g in gamma_range:
        
        svc =SVC(kernel='rbf', gamma=g,C=C)
        
        scores = cross_val_score(svc, x_train, y_trainlabel, cv=5, scoring='accuracy')
       
        b.append((C,g,np.amax(scores)))
        b.sort(key=operator.itemgetter(2),reverse=True)

        
        print("c=",C," gamma=",g," accuracy=",np.amax(scores))
        

print("Optimal hyperparameters (rbf): ","c=",b[0][0]," gamma=",b[0][1]," acc=",b[0][2])


svclassifier = SVC(kernel='rbf',C=b[0][0],gamma=b[0][1])  
start = time.time()
svclassifier.fit(x_train, y_trainlabel) 
end =time.time()
y_pred = svclassifier.predict(x_test)
print("Overall Accuracy (rbf) : ",accuracy_score(y_testlabel,y_pred)*100)
print("Training Time (rbf) :",end-start)
print("----------------------------------------------------------------------------------------------------------------")





'''Poly'''
Css = [2**-5,2**-4,2**-3,2**-2,2**-1,2**1,2**2,2**3,2**4]


gamma_ranges = [2**-6,2**-5,2**-4,2**-3,2**-2] 


b=[]
pol=[1,2,3,4]

for g in gamma_ranges:
    for p in pol:
        for c in Css:
            svc =SVC(kernel='poly',degree=p,C=c,gamma=g)
            
            scores = cross_val_score(svc, x_train, y_trainlabel, cv=5, scoring='accuracy')
            
            b.append((g,p,c,np.amax(scores)))
            b.sort(key=operator.itemgetter(3),reverse=True)
            print("gamma=",g," degree=",p," c=",c," accuracy=",np.amax(scores))
            

print("Optimal hyperparameters (poly) :","gamma=",b[0][0],"degree=",b[0][1],"c=",b[0][2],"accuracy=",b[0][3])



svclassifier = SVC(kernel='poly',C=b[0][2],gamma=b[0][0],degree=b[0][1])  
start = time.time()
svclassifier.fit(x_train, y_trainlabel) 
end =time.time()
y_pred = svclassifier.predict(x_test)
print("Overall Accuracy (poly) : ",accuracy_score(y_testlabel,y_pred)*100)
print("Training Time (poly) :",end-start)
print("----------------------------------------------------------------------------------------------------------------")


''' Linear'''

b=[]

for C in Cs:
    
        
        svc =SVC(kernel='linear',C=C)
        
        scores = cross_val_score(svc, x_train, y_trainlabel, cv=5, scoring='accuracy')
        
        b.append((C,np.amax(scores)))
        b.sort(key=operator.itemgetter(1),reverse=True)

        
        print("c=",C," accuracy=",np.amax(scores))
        

print("Optimal hyperparameters (linear): "," c=",b[0][0]," accuracy=",b[0][1])


svclassifier = SVC(kernel='linear',C=b[0][0])  
start = time.time()
svclassifier.fit(x_train, y_trainlabel) 
end =time.time()
y_pred = svclassifier.predict(x_test)
print("Accuracy (linear) : ",accuracy_score(y_testlabel,y_pred)*100)
print("Training Time (linear) :",end-start)
print("----------------------------------------------------------------------------------------------------------------")

'''Sigmoid'''

b=[]
coeff=[1,0.9,0.8,0.7,-1]

for co in coeff:
    for g in gamma_range:
        
        for c in Cs:
            svc =SVC(kernel='sigmoid', gamma=g,coef0=co,C=c)
            
            scores = cross_val_score(svc, x_train, y_trainlabel, cv=5, scoring='accuracy')
            
            print("c=",c," gamma=",g," coeff=",co," accuracy=",np.amax(scores))
            b.append((co,g,c,np.amax(scores)))
            b.sort(key=operator.itemgetter(3),reverse=True)


print("Optimal hyperparameters (sigmoid):","gamma=",b[0][1],"coef0=",b[0][0],"c=",b[0][2],"accuracy=",b[0][3])


svclassifier = SVC(kernel='sigmoid',C=b[0][2],gamma=b[0][1],coef0=b[0][0])  
start = time.time()
svclassifier.fit(x_train, y_trainlabel) 
end =time.time()
y_pred = svclassifier.predict(x_test)
print("Accuracy (sigmoid) : ",accuracy_score(y_testlabel,y_pred)*100)
print("Training Time (sigmoid) :",end-start)
print("----------------------------------------------------------------------------------------------------------------")
