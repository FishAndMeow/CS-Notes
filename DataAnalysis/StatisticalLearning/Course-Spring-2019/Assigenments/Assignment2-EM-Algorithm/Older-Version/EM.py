# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:47:52 2019

@author: feiyuxiao
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import time

def loadData(libfilepath, vocfilepath):
    print("Loading data......")
    X = []
    #Loading word file
    wordDict = {}
    vocfile = open(vocfilepath,"r")
    for line in vocfile.readlines():
        words = line.split('\t')
        wordDict[int(words[0])] = words[1]
    vocfile.close()
    # Loading lib file
    wordSize = len(wordDict)
    libfile = open(libfilepath,"r")
    for line in libfile.readlines():
        doc = np.zeros(wordSize, dtype='float')
        words = line.strip().split('\t')[1].split(" ")
        for word in words:
            word = word.split(":")
            doc[int(word[0])] = int(word[1])
        X.append(doc)
    libfile.close()
    return np.asarray(X), wordDict

def preprocessor(datasetname,vocabname):

    f = open(vocabname, "r")  
    voc = []
    lines = f.readlines()  
    
    for line in lines: 
        vocline=[]
        for word in line.split():
             vocline.append(word)
        vocline[0] = int(vocline[0])
        vocline[2] = int(vocline[2])
        voc.append(vocline)
    f.close()
    
    W = len(voc)
    
    F = open(datasetname, "r")  
    Lines = F.readlines()  
    Doc_label=[]
    Doc = []
    for line in Lines: 
        Doc_label.append(int(line.split()[0]))
        d = []
        for word in line.split()[1:]:
            d.append(word.split(":"))  
        Doc.append(d)
    F.close()
    
    D = len(Doc_label)
    
    T = np.zeros((D,W))
    for i in range(D):
        for j in range(len(Doc[i])):
            T[i][int(Doc[i][j][0])] = int(Doc[i][j][1])
    
    return voc,T

def printResults(mu_log, wordDict,NumMostFrequent=10):
    U = mu_log.transpose()
    (K,W)=U.shape
    print("Ushape",U.shape)
    #NumMostFrequent =5
    word = [[] for i in range(K)]
    #f = file('./nips/output.txt', 'w+')
    print("The K is: %d" %(K))
    for i in range(K):
        #print "The cluster %d\'s ratio is %f, most-frequent %d words are: " % (i+1, self.PI[i], num)
        #print>>f, "The cluster %d\'s ratio is %f, most-frequent %d words are: " % (i+1, self.PI[i], num)
        list = []
        for j in range(W):
            list.append((U[i,j],j))
        list.sort(key=lambda x:x[0], reverse=True)
        print("-"*50)
        print('Frequent words of topic {}:'.format(i))
        for j in range(NumMostFrequent):
            word[i].append(wordDict[list[j][1]])
            #print(wordDict[list[j][1]])
            #print>>f, dataset.voc_dict[list[j][1]]
        print(word[i])
        print("")
      
        
def Frequent_words(mu_log, vocabulary, Num=10):
    """
    args:
        log_mu: numpy array of shape (num_words, num_topics)
    """
    topics_number = mu_log.shape[1]
    mu = np.exp(mu_log)
    print(mu.shape)
    frequent_words_list = np.argpartition(mu, -Num, axis=0)[-Num:]
    print("frequent_words_list[doc]")
    print(frequent_words_list.shape)
    for doc in range(topics_number):
        print("-"*50)
        print('Frequent words of topic {}:'.format(doc))
        print(frequent_words_list[doc])
        words = [voc[id][1] for id in  frequent_words_list[doc]]
        print(' '.join(words))

def freqvisual(T):
    
    freq = T.sum(axis=0)
    
    plt.figure()
    plt.plot(freq)  
    plt.show()
    
    numbermin = 20
    
    obj = []
    
    for i in range(len(freq)):
        if freq[i]<numbermin:
            obj.append(i)
            
    T=np.delete(T, obj, axis=1)
    Freq = T.sum(axis=0)
    
    plt.figure()
    plt.plot(Freq)  
    plt.show()
    
    return T,obj

def Normalize_array(array):
    sum_array = logsumexp(array)
    return array - np.array([sum_array])

def Normalize_matrix(matrix,axis):
    '''
    Matrix
    '''
    sum_matrix = logsumexp(matrix,axis=axis)
    if sum_matrix.shape[0] == matrix.shape[0]:
        return (matrix.transpose() - sum_matrix).transpose()
    else:
        return matrix - sum_matrix

class EM():
    '''
        word_matrix: word occurrence matrix for the corpus
    '''
    def __init__(self,topics_number = 10,error = 0.0000001):
        self.topics_number = topics_number
        self.pi = None
        self.mu = None
        self.er = error
    
    def cal_likehood(self):
        '''
        gamma_un_log: log of gamma (Not uniform)
        And this matrix is D * K
        '''
        word_matrix = self.word_matrix
        D,W = word_matrix.shape
        pi_log = self.pi_log
        mu_log = self.mu_log
        gamma_un_log = np.dot(word_matrix,mu_log) + pi_log 
        likehood = logsumexp(gamma_un_log,axis=1).sum()
        return likehood
    
    
    def initialize(self):
        word_matrix = self.word_matrix
        topics_number = self.topics_number
        W = word_matrix.shape[1]
        '''
        pi_log = -np.log(topics_number)*np.ones(topics_number)
        Tw = word_matrix.sum(axis=0)
        mu_log_k = np.log(Tw) - np.log(Tw.sum())    
        mu_log = np.repeat(mu_log_k.reshape(-1, 1), topics_number, axis=1)
      
        
        '''
        pi = np.random.randint(1, 9, size=topics_number)
        pi_log = np.log(pi) - np.log(pi.sum())
        mu = np.random.randint(1, 9, size=(W, topics_number))
        mu_log = np.log(mu) - np.log(mu.sum(axis=0))
  
        self.pi_log = pi_log
        self.mu_log = mu_log
        
    def Expectation(self):
        '''
        gamma_uniform_log: gamma_log which is uniform
        '''
        word_matrix = self.word_matrix
        pi_log = self.pi_log
        mu_log = self.mu_log
        gamma_un_log = np.dot(word_matrix,mu_log) + pi_log 
        gamma_uniform_log = Normalize_matrix(gamma_un_log,axis=1)
        
        self.gamma_log = gamma_uniform_log
        
    def Maximization(self):
        word_matrix = self.word_matrix
        gamma_log = self.gamma_log
        topics_number = self.topics_number
        D,W = word_matrix.shape[0],word_matrix.shape[1]
        mu_un_log = logsumexp(gamma_log.reshape(D,topics_number,1),
                              b=word_matrix.reshape(D,1,W),axis = 0).transpose() # !!!!
        mu_uniform_log = Normalize_matrix(mu_un_log,axis=0)
        
        pi_un_log = logsumexp(gamma_log,axis=0)
        pi_uniform_log = Normalize_array(pi_un_log)
        
        self.pi_log = pi_uniform_log
        self.mu_log = mu_uniform_log
    
    def main(self,word_matrix):
        self.word_matrix = word_matrix
        self.initialize()
        likehood_old =  self.cal_likehood()
        likehood_change = np.infty
        iteration = 0
        #Nd = np.sum(self.word_matrix, axis=0)
        print("main")
        print('Tolerance',self.er)
        #print(likehood_change)
        #pi_log = None
        mu_log = None
        while likehood_change >= self.er:
            self.Expectation()
            self.Maximization()
            #pi_log = self.pi_log
            mu_log = self.mu_log
            #print("shape",mu_log.shape)
            likehood_new = self.cal_likehood()
            likehood_change = likehood_new - likehood_old
            likehood_old = likehood_new
            
            iteration +=1
            #mu = np.exp(mu_log)
            #U = (self.word_matrix.dot(mu) / np.sum(mu.transpose() * Nd, axis=1)).transpose()
            print("training",iteration,likehood_change)
        #Frequent_words(mu_log, voc)
        #visual_frequent_words(mu_log,voc)
        
        printResults(mu_log, wordDict)
        print("Finished")
        
filename1 = 'nips/nips.libsvm'
filename2 = "nips/nips.vocab"
voc,T = preprocessor(filename1,filename2)
X, wordDict = loadData(filename1,filename2)
time_start=time.time()
trainingmethod = EM()
trainingmethod.main(T)
time_end=time.time()
print('Totally Time Cost',time_end-time_start)

