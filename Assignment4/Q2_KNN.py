# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:35:40 2018

@author: Susan
"""
import numpy as np
from collections import Counter
from sklearn import datasets

def KNN(test,train,target,K):
    N = train.shape[0]
    dist = np.sum((np.tile(test,(N,1))-train)**2,axis=1) 
    idx = sorted(range(len(dist)),key=lambda i:dist[i])[0:K] 
    
    return Counter(target[idx]).most_common(1)[0][0]

iris = datasets.load_iris()
X = iris.data
target = iris.target

N = X.shape[0]

CF_array = []
for j in range(10):
    CF = np.zeros((3,3))
    for i in range(N):
        train_idx = np.setdiff1d(np.arange(N),i)
        guess = KNN(X[i,:],X[train_idx,:],target[train_idx],j+1)
        CF[target[i],guess] += 1
    CF_array.append(CF)
    print(CF)
