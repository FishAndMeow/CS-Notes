# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:47:52 2019

@author: feiyuxiao
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

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


def Frequent_words(log_mu, vocabulary, topN=20):
    """
    args:
        log_mu: numpy array of shape (num_words, num_topics)
    """
    num_words, num_mixtures = log_mu.shape
    top_words_idx = np.argpartition(log_mu, -topN, axis=0)[-topN:]
    print(top_words_idx)
    print(top_words_idx.shape)
    for topic_id in range(num_mixtures):
        print("-"*80)
        print('Frequent words of topic {}:'.format(topic_id))
        words = [voc[word_idx][1] for word_idx in top_words_idx[topic_id]]
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
    def __init__(self,topics_number = 20,error = 0.1):
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
            likehood_new = self.cal_likehood()
            likehood_change = likehood_new - likehood_old
            likehood_old = likehood_new
            
            iteration +=1
            
            print("training",iteration,likehood_change)
        visual_frequent_words(mu_log, voc, topN=20)
        print("Finished")

"""
Main 
"""           
        
filename1 = 'nips/nips.libsvm'
filename2 = "nips/nips.vocab"
voc,T = preprocessor(filename1,filename2)

trainingmethod = EM()
trainingmethod.main(T)
