# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 11:11:17 2018

@author: Susan
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black Shoes 
def blsprice(S,L,T,r,vol):
    d1 = (math.log(S/L)+(r+0.5*vol**2)*T)/(vol*math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)
    call = S * norm.cdf(d1) - L * math.exp(-r*T) * norm.cdf(d2)
    return call

# Monte Carlo method
def MCsim(S,T,r,vol,N):
    dt = T/N
    St = np.zeros((N+1))
    St[0] = S
    for i in range(N):
        St[i+1]= St[i]*math.exp((r-0.5*vol**2)*dt+np.random.normal()*vol*math.sqrt(dt))
    return St

# Use Bisection black shoe (similar to binary search)
def BisectionBLS(S,L,T,r,call):
    left = 0.00000001
    right = 1
    # error
    while(right - left > 0.00000001):
       middle = (left+right)/2
       if ((blsprice(S,L,T,r,middle)-call)*(blsprice(S,L,T,r,left)-call)) < 0:
           right = middle
       else:
           left = middle
    return (left+right)/2

# Newton BLS
def NewtonBLS(S,L,T,r,call):
    vol = 0.5
    while(abs(blsprice(S,L,T,r,vol)-call) > 0.00001):
        vol = vol-(blsprice(S,L,T,r,vol)-call)/((blsprice(S,L,T,r,vol + 0.00001)-blsprice(S,L,T,r,vol-0.00001))/0.00002)
    return vol

#%%
# initiation
S = 50
L = 40
T = 2
r = 0.08
vol = 0.2

#%%
# Q1: 20000 times of simulation by Monte Carlo method 
N = 100
M = 20000
counter=0
Sa200 = np.zeros(200)
Sa2000 = np.zeros(2000)

call = 0
trend = []
for i in range(M):
    Sa = MCsim(S,T,r,vol,N)
    trend.append(Sa[-1])

plt.hist(trend[:200], bins=100)
plt.show()
plt.hist(trend[:2000], bins=100)
plt.show()
plt.hist(trend, bins=300)
plt.show()
#%%
# Q2： calculate the call price and loss between BLS v.s. MC
N1 = 100
N2 = 1000
N3 = 10000
L = 40
eps = np.random.normal(0,1,(1000,100))
prod_eps = np.prod(eps,axis=1)

Sn = np.mean(S*np.exp((r-0.5*vol**2)*(T/N1) + prod_eps*vol*np.sqrt(T/N1)) - L ) * np.exp(-r*T)

#%%
# Q3-1: calculate the vol by Bisection & Newton
# initial
S = 10978.85
L = 11200
call = 40.5
r = 0.0109
T = 22/365
vol = 0.5
left = 0.00000001
right = 1

# list for plot
Bi,Newton =[],[]
x = [i for i in range(1,21)]

# calculate 20 times
for i in range(0,20):
    # Bisection method
    if(right - left > 0.00000001):
       middle = (left+right)/2
       if ((blsprice(S,L,T,r,middle)-call)*(blsprice(S,L,T,r,left)-call)) < 0:
           right = middle
       else:
           left = middle
    Bi.append((left+right)/2)
    
    # Newton method
    if(abs(blsprice(S,L,T,r,vol)-call) > 0.00001):
        vol = vol-(blsprice(S,L,T,r,vol)-call)/((blsprice(S,L,T,r,vol + 0.00001)-blsprice(S,L,T,r,vol-0.00001))/0.00002)
    Newton.append(vol)

# plot the answer
plt.xticks(x)
plt.xlabel('iteration times')
plt.ylabel('volatility') 
plt.plot(x,Bi)
plt.plot(x, Newton)    
plt.show()
#%%
# Q3-2: the volatility from call price 10900-11600
S = 10978.85
L = [10900,11000,11100,11200,11300,11400,11500,11600]
call = [173,115,71,40.5,22.5,11.5,6.4,3.4]
r = 0.0109
T = 22/365
Vol = []

for i in range(0,8):
    Li = L[i]
    calli = call[i]    
    Vol.append(NewtonBLS(S,Li,T,r,calli))

# plot the answer
plt.xticks(L)
plt.xlabel('L(Exercise Price)')
plt.ylabel('volatility') 
plt.plot(L, Vol)    
plt.show()
