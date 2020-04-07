# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 21:09:50 2018

@author: Susan
"""

import numpy as np
import pandas as pd
import math
from sklearn import datasets

# p = number of positive ; n = number of negative   
def entropy(p1,n1):
    if(p1==0 and n1==0):
        return 1 
    elif(p1==0 or n1==0):
        return 0
    pp =  p1/(p1+n1) #positive probability
    np = n1/(p1+n1) #negative probability
    return -pp*math.log2(pp)-np*math.log2(np)

# calculate infomation Gain
def infoGain(p1,n1,p2,n2):
    total = p1+p2+n1+n2 # number of total data
    num_c1 = p1+n1    # number of class1
    num_c2 = p2+n2    # number of class2
    return entropy(p1+p2,n1+n2)-num_c1/total*entropy(p1,n1)-num_c2/total*entropy(p2,n2)

def BuildTree(c1,c2,x,y):
    node = dict()
    node['data'] = np.arange(len(y),dtype=int)
    Tree = []
    Tree.append(node)

    t = 0
    while(t<len(Tree)):
        idx = Tree[t]['data']
        if (len(np.unique(y[idx]))==1):
            Tree[t]['leaf']=1    
            Tree[t]['decision']= np.unique(y[idx])[0]
        else:
            bestIG = 0
            for i in range(x.shape[1]):
                pool = list(set(x[idx,i]))
                pool.sort()
                for j in range(len(pool)-1):
                    thres = (pool[j]+pool[j+1])/2
                    G1 = []
                    G2 = []
                    for k in idx:
                        if(x[k][i]<=thres):
                            G1.append(k)
                        else:
                            G2.append(k)
                    thisIG = infoGain(sum(y[G1]==c2),sum(y[G1]==c1),sum(y[G2]==c2),sum(y[G2]==c1))
                    if(thisIG>bestIG):
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 = G2
                        bestthres = thres
                        bestf = i
            if(bestIG>0):
                Tree[t]['leaf'] = 0
                Tree[t]['selectf'] = bestf
                Tree[t]['threshold'] = bestthres
                Tree[t]['child'] = [len(Tree),len(Tree)+1]
                node = dict()
                node['data'] = bestG1
                Tree.append(node)
                node = dict()
                node['data'] = bestG2
                Tree.append(node)
            else:
                Tree[t]['leaf'] = 1
                if(sum(y[idx]==c2)>sum(y[idx]==c1)):
                    Tree[t]['decision']= c2
                else:
                    Tree[t]['decision']= c1
                
        t = t+1
    return Tree
#%%
def ans(Tree,feature):
    ans = []
    for test_feature in feature:
        now = 0
        while(Tree[now]['leaf']==0):
            bestf = Tree[now]['selectf']
            thres = Tree[now]['threshold']
            if(test_feature[bestf]<=thres):
                now = Tree[now]['child'][0]
            else:
                now = Tree[now]['child'][1]
        ans.append(Tree[now]['decision'])
        
    return ans
#%%
iris = datasets.load_iris()
A = []
error = []
group = np.mod(np.arange(150),50)//5
for testg in range(10):
    A=[]

    trainidx = np.where(group!=testg)[0] 
    testidx = np.where(group==testg)[0]
    
    tree01 = BuildTree(0,1,iris.data[trainidx],iris.target[trainidx])
    tree02 = BuildTree(0,2,iris.data[trainidx],iris.target[trainidx])
    tree12 = BuildTree(1,2,iris.data[trainidx],iris.target[trainidx])
    
    t1 = ans(tree01,iris.data[testidx])
    t2 = ans(tree02,iris.data[testidx])
    t3 = ans(tree12,iris.data[testidx])   

    for i in range(len(t1)):
        if t1[i]==t2[i]==t3[i]:
            A.append(t1[i])
        elif (t1[i]!=t2[i] and t2[i]!=t3[i] and t1[i]!=t3[i]):
            A.append(0)
        else:
            if t1[i]==t2[i]:
                A.append(t1[i])
            elif t1[i]==t3[i]:
                A.append(t1[i])
            else:
                A.append(t2[i])
    error.append(np.count_nonzero(A-iris.target[testidx])/len(iris.target[testidx]))
error_rate = np.mean(error)
print('Error rate:%f' % error_rate)