# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:51:05 2018

@author: Susan
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 


def F1(t):
    # noise：np.random.normal(0,1)
    return 0.063*(t**3) - 5.284*(t**2) + 4.887*t + 412 + np.random.normal(0,1)

def F2(t,A,B,C,D):
    return A*(t**B)+C*np.cos(D*t)+np.random.normal(0,1,t.shape)

def Energy(b2,T,A,B,C,D):
    return np.sum(abs(b2-F2(T,A,B,C,D)))

def P_predict(A,B,C,tc,beta,omega,fy):
    t = np.arange(0,tc,1)
    return A+(B*(tc-t)**beta)*(1+C*np.cos(omega*np.log(tc-t)+fy))

def Energy_p(A,B,C,tc,beta,omega,fy):
    return np.sum(abs(ln_data[0:P_predict(A,B,C,tc,beta,omega,fy).shape[-1]]-P_predict(A,B,C,tc,beta,omega,fy)))
#%%
n = 1000
A = np.zeros((n,5))
b = np.zeros((n,1))

for i in range(n):
    t = np.random.random()*100
    b[i] = F1(t)
    A[i,0] = t**4
    A[i,1] = t**3
    A[i,2] = t**2
    A[i,3] = t
    A[i,4] = 1

x = np.linalg.lstsq(A,b)[0]
print(x)
T = np.random.random((n,1))*100
b2 = F2(T,0.6,1.2,100,0.4)
#%%
# Q1 plot the curve
Ans = []
D1 = np.arange(-5.11,5.12,0.01)        
for i in range(len(D1)):
    Ans.append(Energy(b2,T,0.6,1.2,100,D1[i]))

plt.plot(Ans)

#%%
# Q2 plot the surface 
Ans2 = np.zeros((1023,1023))
A2 = np.arange(-5.11,5.12,0.01)
C2 = np.arange(-511,512,1)

# making data
for i in range(len(A2)):
    for j in range(len(C2)): 
        Ans2[i][j] = Energy(b2,T,A2[i],1.2,C2[j],0.4)

# Plot the surface
X, Y = np.meshgrid(A2, C2)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Ans2, cmap=plt.get_cmap('rainbow'))
plt.show()
#%% 
# Q3 LPPL
data = np.load('HW5_data.npy')
ln_data = np.log(data)

'''Linear Regression：求ABC'''
beta = 0.5
omega = 3
tc = 505
fy = 5
t = np.arange(0,tc,1)

for k in range(30):
    大矩陣 =np.ones((tc,3))
    b = ln_data[0:tc]
    大矩陣[:,1] = (tc-t)**beta
    大矩陣[:,2] = 大矩陣[:,1]*np.cos((omega*np.log(tc-t))+fy)
    
    x = np.linalg.lstsq(大矩陣,b)[0]
    
    '''Gene Algorithm：求超參數'''
    pop = np.random.randint(0,2,(10000,16))
    fit = np.zeros((10000,1))
    
    A = x[0]
    B = x[1]
    C = x[2]/x[1]
    
    for generation in range(10):    # range寫10：繁衍十代
        for i in range(10000):      # 10000人中，留前100名活下
            gene = pop[i,:]
            tc = np.sum(2**np.arange(4)*gene[0:4])+500
            beta = np.clip(np.sum(2**np.arange(4)*gene[4:8])/16, 0.000001, 0.9999999)
            omega = np.sum(2**np.arange(4)*gene[8:12])
            fy = np.clip((np.sum(2**np.arange(4)*gene[12:16]))*np.pi/8, 0.000001, np.pi*2-0.000001)
            fit[i] = Energy_p(A,B,C,tc,beta,omega,fy)
        sortf = np.argsort(fit[:,0])    # 用argsort會依由大到小排序並取出index
        pop = pop[sortf,:]              
        for i in range(100,10000):  # 把100名以後的人，全部用前100名交配產生
            fid = np.random.randint(0,100)  # 從100人當中尋找 father ID
            mid = np.random.randint(0,100)  # 從100人當中尋找 mother ID
            while mid == fid :               # 若父母同一人，則媽媽重找
                mid = np.random.randint(0,100)
            mask = np.random.randint(0,2,(1,16))    # 16代表16個基因
            son = pop[mid,:]        # 兒子先複製媽媽的
            father = pop[fid,:]
            son[mask[0,:]==1] = father[mask[0,:]==1]    # 如果mask等於1，則換成爸爸的
            pop[i,:] = son  # 把son assign 回population
        for i in range(1000):                   # 基因突變的部分
            m = np.random.randint(0,10000)      # 取出第m個人的第n個基因突變
            n = np.random.randint(0,16)
            pop[m,n] = 1-pop[m,n]               # 相減以產生1變0，0變1的狀況
    for i in range(10000):      # 再適者生存：10000人中，留前100名活下
        gene = pop[i,:]
        tc = np.sum(2**np.arange(4)*gene[0:4])+500
        beta = np.clip(np.sum(2**np.arange(4)*gene[4:8])/16, 0.000001, 0.9999999)
        omega = np.sum(2**np.arange(4)*gene[8:12])
        fy = np.clip((np.sum(2**np.arange(4)*gene[12:16]))*np.pi/8, 0.000001, np.pi*2-0.000001)
        fit[i] = Energy_p(A,B,C,tc,beta,omega,fy)
    sortf = np.argsort(fit[:,0])    # 用argsort會依由大到小排序並取出index
    pop = pop[sortf,:]
    
    
    gene = pop[0,:]
    tc = np.sum(2**np.arange(4)*gene[0:4])+500
    beta = np.clip(np.sum(2**np.arange(4)*gene[4:8])/16, 0.000001, 0.9999999)
    omega = np.sum(2**np.arange(4)*gene[8:12])
    fy = np.clip((np.sum(2**np.arange(4)*gene[12:16]))*np.pi/8, 0.000001, np.pi*2-0.000001)
    t = np.arange(0,tc,1)

plt.plot(ln_data)
plt.plot(P_predict(A,B,C,tc,beta,omega,fy))
