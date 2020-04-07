# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:44:03 2018

@author: Susan
"""
import matplotlib.pyplot as plt
import numpy as np

call = {10600:418,10700:332,10800:245,10900:173,11000:115,11100:71,11200:40.5}
put = {10600:24.5,10700:37,10800:57,10900:86,11000:127,11100:183,11200:251}
x = np.arange(10500,11501)

# the profit of a call
def callr(K):
    global x
    global call
    return np.maximum(x-K,0)-call[K]

def putr(K):
    global x 
    global put
    return np.maximum(K-x,0)-put[K]

# Q1:Bull Spread
y1 = callr(10800)-callr(10900)
y2 = callr(10800)-callr(11000)
y3 = callr(10900)-callr(11000)

plt.plot(x, y1, 'r', x, y2, 'g',x, y3,  x, np.zeros((len(x))), '--')
plt.show()

# Q2:Straddle vs. Stangle
# A：Straddle of 10900
y1 = -callr(10900) - putr(10900) # Sale
y2 = callr(10900) + putr(10900)  # Buy
plt.plot(x, y1, 'r',x,y2,'g', x, np.zeros((len(x),)), '--')
plt.show()

# B：y1 = Straddle vs. y2 = Strangle
y1 = -callr(10900) - putr(10900)
y2 = -callr(11000) - putr(10800)

plt.plot(x, y1, 'r', x, y2, 'g', x, np.zeros((len(x),)), '--')
plt.show()

# Q3:Arbitrage space
y1 = putr(11200)-callr(11200)
y2 = putr(10600)-callr(10600)

plt.plot(x, y1, 'r', x, y2, 'g', x, np.zeros((len(x),)), '--')
plt.show()

# Q4:Butterfly Spread (yellow line)
y1 = callr(10900)
y2 = callr(11100)
y3 = -callr(11000) *2
y4 = y1+y2+y3
plt.plot(x, y1, 'r', x, y2, 'g', x, y3, 'b', x, y4, 'y', x, np.zeros((len(x),)), '--')
plt.show()
