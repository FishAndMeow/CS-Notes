# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 18:43:06 2019

@author: feiyuxiao
"""

import numpy as np
from sklearn.datasets import load_svmlight_file


def loadData(filename):
    X,Y = load_svmlight_file(filename)
    X = X.todense()
    Y = (Y+1)/2
    for y in Y:
        y = round(y)   # to resize to {0,1}      
    return X,Y


def sigmoid(x):
    from numpy import exp
    if x>=0:
        return 1.0/(1+exp(-x))
    else:
        return exp(x)/(1+exp(x))

def SGD (dataX,datalabel):
    m,n = np.shape(dataX)
    alpha = 0.001 # learning rate
    max_iter = 50 # max iteration cycles
    ratios = np.ones(n) # the weights
    
    for x in range(max_iter):
        for i in range(m):
            mu = sigmoid(ratios*dataX[i].T)
            error = datalabel[i] - mu
            
            ratios  = ratios +  alpha*error*dataX[i]
            '''
            if x == 0 and i ==0:
                results = ratios.T
                print(results.shape)
            else:
                results = np.stack([results,ratios.T])
            '''
    return ratios #,results

file_train = "a9a/a9a"
file_test = "a9a/a9a.t"


X_train , y_train = loadData(file_train)
X_test , y_test = loadData(file_test)

row_xtest = X_test.shape[0]

line = np.zeros(row_xtest)

X_test = np.column_stack((X_test,line))

print(X_test.shape)

weights = SGD(X_train,y_train)

ytrain_size = y_train.shape[0]
ytest_size = y_test.shape[0]


y_train_pre = np.zeros(ytrain_size)

y_test_pre = np.zeros(ytest_size)

accuracy_score_train = 0
accuracy_score_test = 0

for i in range(ytrain_size):
    y_train_pre[i] = sigmoid(weights*X_train[i].T)
    if (abs(y_train_pre[i]-y_train[i]) < 0.5):
        accuracy_score_train+=1

accuracy_score_train = accuracy_score_train/ytrain_size


for i in range(ytest_size):
    y_test_pre[i] =  sigmoid(weights*X_test[i].T)
    if (abs(y_test_pre[i]-y_test[i]) < 0.5):
        accuracy_score_test+=1

accuracy_score_test = accuracy_score_test/ytest_size        
  
print("finished")
print(accuracy_score_train)  
print(accuracy_score_test)