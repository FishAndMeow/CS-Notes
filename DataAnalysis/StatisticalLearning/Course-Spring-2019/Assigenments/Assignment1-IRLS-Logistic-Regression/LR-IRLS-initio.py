# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 18:43:06 2019

@author: feiyuxiao
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
import scipy.io as sio


import matlab.engine
eng = matlab.engine.start_matlab()


def loadData(filename):
    X,Y = load_svmlight_file(filename)
    X = X.todense()
    X = np.array(X)
    Y = (Y+1)/2
    for y in Y:
        y = round(y)   # to resize to {0,1}  
    #print("data load",type(X),type(Y))
    return X,Y


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
    
def SparseMatrixMultiply(A, B):#减少计算次数
    AA = csr_matrix(A)
   # BB = csr_matrix(B)
    res = AA.dot(B)
    return res

def diagMatrixMultiply(A,D): # D 是对角阵
    m = A.shape[1]
    for i in range(m):
        A[:,i] =  A[:,i]*D[i]
    return A

def simplymultiply(A,B):
    C = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        C[i] = A[i]*B[i]
    return C


def IRLS (dataX,datalabel,epsilon = 0.00001,max_iter = 30):
    m,n = np.shape(dataX)
       
    ratios = np.zeros(n)
    ratios_update = np.zeros(n)
    
    times_iter = 0
    
    for x in range(max_iter):
        
        temp0 =dataX.dot(ratios)
        Y = sigmoid(temp0)
        
        R = Y*(1.0-Y)
       
        Rnn = np.diag(R)
        
        matrix0 = dataX.transpose().dot(Rnn)
        temp1 = matrix0.dot(dataX)
        
        #matrix1 = np.linalg.inv(temp1)
        
        temp2 = temp0+ (datalabel-Y)/R
       
        #ratios_update = matrix1.dot(matrix0).dot(temp2)
        # Use matlab to solve the ill contional equation
        temp3 = matrix0.dot(temp2)
        sio.savemat("equation.mat",{"A":temp1,"b":temp3})
        double_ratios = eng.matsolve()
        double_ratios = np.array(double_ratios)
        
        ratios_update[:] =  double_ratios[:,0]
        
        
        print("ratios",ratios_update.shape)
        error = ratios_update - ratios
        error_norm = np.linalg.norm(error)
        ratios = ratios_update
        print("error",error_norm)
        if error_norm < epsilon:
            break
        ratios = ratios_update    
        
        times_iter += 1
    return ratios,times_iter #,results
   
def accuray(dataX,datalabel,weights_pre):
    y_pre = np.zeros(datalabel.shape[0])
    y_pre = sigmoid(dataX.dot(weights_pre))
    flag = 0
    for i in range(datalabel.shape[0]):
        if (abs(y_pre[i]-datalabel[i]) < 0.5):
            flag+=1

    accuracy_score = flag/datalabel.shape[0]
   
    return accuracy_score    

# main
#file_train = "a9a"
file_train = "a9a/a9a"
file_test = "a9a/a9a.t"


X_train , y_train = loadData(file_train)
X_test , y_test = loadData(file_test)

row_xtest = X_test.shape[0]

line = np.zeros(row_xtest)

X_test = np.column_stack((X_test,line))

#lambda_list = [0.01,0.05,0.1,0.2]
lambda_list = [0.01]
ratios = []
accuracy_train = []
accuracy_test = []
time = []
epsilon = 0.1
max_iter = 20

#for Lambda in lambda_list:    
ratio,_time = IRLS(X_train,y_train)
ratios.append(ratio)
time.append(_time)
accuracy_score_train = accuray(X_train,y_train,ratio)
accuracy_train.append(accuracy_score_train)
accuracy_score_test = accuray(X_test,y_test,ratio)       
accuracy_test.append(accuracy_score_test)

chioce_Lambda = accuracy_test.index(max(accuracy_test)) 
  
print("finished")
#print('iteration times',times)
print("times_iter",_time)
print(accuracy_train[chioce_Lambda])  
print(accuracy_test[chioce_Lambda])

plt.figure()
plt.plot(lambda_list,time)

plt.figure()
plt.plot(lambda_list,accuracy_train)

plt.figure()
plt.plot(lambda_list,accuracy_test)
