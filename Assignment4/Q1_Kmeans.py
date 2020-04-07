# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:19:14 2018

@author: Susan
"""
import random
import numpy as np
from sklearn import datasets

def kmeans(sample,K,maxiter):
    N = sample.shape[0]
    idx = random.sample(range(N),K)
    C = sample[idx,:]
    L = np.zeros((N,1))
    dist = np.zeros((N,K))
    iter = 0
    while(iter<maxiter):
        for i in range(K):  
            dist[:,i] = np.sum((sample-np.tile(C[i],(N,1)))**2,1)
        L1 = np.argmin(dist,1) 
        if(iter>0 and np.array_equal(L,L1)):
            break
        L = L1
        for i in range(K):
            idx = np.nonzero(L==i)[0]   
            if(len(idx)>0):             
                C[i,:] = np.mean(sample[idx,:],0)
        iter += 1
    wicd = np.mean(np.sqrt(np.sum((sample-C[L,:])**2,axis=1))) 
    return C,L,wicd


iris = datasets.load_iris()
X = iris.data
X1 = (X-np.tile(X.mean(0),(X.shape[0],1)))/np.tile(X.std(0),(X.shape[0],1))          
X2 = (X-np.tile(X.min(0),(X.shape[0],1)))/(X.max(0)-np.tile(X.min(0),(X.shape[0],1)))

C1,L1,wicd1 = kmeans(X1,3,1000)
C2,L2,wicd2 = kmeans(X2,3,1000)

print("wicd1: %f\nwicd2: %f" %(wicd1,wicd2))