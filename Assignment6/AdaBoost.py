# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 20:24:46 2018

@author:Susan rsps971130
"""
# implement pepper：人臉辨識
import math
import numpy as np
import matplotlib.pyplot as plt

# load the photo array
npzfile = np.load('1114_CBCL.npz')
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
testface = npzfile['arr_2']
testnonface = npzfile['arr_3']

trpn = trainface.shape[0]        # train positive number
trnn = trainnonface.shape[0]     # train non face number
tepn = testface.shape[0]         # test positive number
tenn = testnonface.shape[0]      # test non face number

# make sure we load the right data (plot out one photo)
face1 = trainface[15,:].reshape((19,19))
plt.imshow(face1,cmap='gray')

# 窮舉各種 Feature 排列組合(各種不同的長方型)
ftable = []     # feature
fn = 0          # total feature number：36648

for y in range(19):     # for y in image width&high
    for x in range(19): # Rectagle的長寬最小從2開始(因為1就是一條線而已)，最大可以到20
        for h in range(2,20):
            for w in range(2,20):
                '''確認取出的rectangle介於整張圖片19*19內，即可視為合理特徵'''
                # featureA(左右) 數目：12312 
                if y+h-1<=18 and x+w*2-1<=18:
                    fn +=1
                    ftable.append([0,y,x,h,w])
                    
                # featureB(上下) 數目：12312
                if y+h*2-1<=18 and x+w-1<=18:
                    fn +=1
                    ftable.append([1,y,x,h,w])
                    
                # featureC(左中右) 數目：6840
                if y+h-1<=18 and x+w*3-1<=18:
                    fn +=1
                    ftable.append([2,y,x,h,w])

                # featureD(左上右下) 數目：5184
                if y+h*2-1<=18 and x+w*2-1<=18:
                    fn +=1
                    ftable.append([3,y,x,h,w])
                    
    
def fe(sample,ftable,c):    # 計算 feature格子內的差
    ftype,y,x,h,w = ftable[c][:]
    T = np.arange(361).reshape((19,19))
    if(ftype==0):
        zone1 = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)
        zone2 = np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis=1)
        return zone1 - zone2
    elif(ftype==1):
        zone1 = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)
        zone2 = np.sum(sample[:,T[y+h:y+h*2,x:x+w].flatten()],axis=1)
        return zone1 - zone2
    elif(ftype==2):
        zone1 = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)
        zone2 = np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis=1)
        zone3 = np.sum(sample[:,T[y:y+h,x+w*2:x+w*3].flatten()],axis=1)
        return zone1 + zone3 - zone2
    else:
        zone1 = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)
        zone2 = np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis=1)
        zone3 = np.sum(sample[:,T[y+h:y+h*2,x:x+w].flatten()],axis=1)
        zone4 = np.sum(sample[:,T[y+h:y+h*2,x+w:x+w*2].flatten()],axis=1)
        return zone1 + zone4 - zone2 -zone3

# weak classifier
def WC(pw,nw,pf,nf):
    maxf = max(pf.max(),nf.max())
    minf = min(pf.min(),nf.min())
    theta = (maxf-minf)/10 + minf
    polarity = 1    # 左邊的人是正的，右為負
    error = np.sum(nw[nf<theta]) + np.sum(pw[pf>=theta])
    if(error>0.5):  # 如果error大於0.5直接把p反向，先假設第一刀為最好的一刀
        polarity = -1
        error = 1 - error
    min_theta,min_polarity,min_error = theta,polarity,error
    for i in range(2,10): # 再切接下來的2-10刀
        theta = (maxf-minf)*i/10 + minf
        polarity = 1
        error = np.sum(nw[nf<theta]) + np.sum(pw[pf>=theta])
        if(error>0.5): # 若error>0.5，p反向設置
            polarity = -1
            error = 1 - error
        if(error<min_error): 
            min_theta,min_polarity,min_error = theta,polarity,error
    return min_error,min_theta,min_polarity
#%%
# train
trpf = np.zeros((trpn,fn))
trnf = np.zeros((trnn,fn))
for c in range(fn):
    trpf[:,c] = fe(trainface,ftable,c)
    trnf[:,c] = fe(trainnonface,ftable,c)

# test
tepf = np.zeros((tepn,fn))
tenf = np.zeros((tenn,fn))
for c in range(fn):
    tepf[:,c] = fe(testface,ftable,c)
    tenf[:,c] = fe(testnonface,ftable,c)

pw = np.ones((trpn,1))/trpn/2
nw = np.ones((trnn,1))/trnn/2
#%%
# t個分類器，存在SC裡面
SC = []
for t in range(50):
    weightsum = np.sum(pw)+np.sum(nw)
    pw = pw/weightsum
    nw = nw/weightsum
    best_error,best_theta,best_polarity = WC(pw,nw,trpf[:,0],trnf[:,0])
    best_feature = 0
    for i in range(1,fn):
        error,theta,polarity = WC(pw,nw,trpf[:,i],trnf[:,i])
        if(error<best_error):
            best_feature = i
            best_error,best_theta,best_polarity = error,theta,polarity
    beta = best_error/(1-best_error)
    alpha = math.log10(1/beta)
    SC.append([best_feature,best_theta,best_polarity,best_error,alpha])
    if(best_polarity==1):
        pw[trpf[:,best_feature]<best_theta] *= beta
        nw[trnf[:,best_feature]>=best_theta] *= beta
    else:
        pw[trpf[:,best_feature]>=best_theta] *= beta
        nw[trnf[:,best_feature]<best_theta] *= beta
#%%
#==================================================
#     Q1 : draw the ROC curve of test & train  
#==================================================
N = [3,5,10,20,50]
def ROC_curve(N,trpf,trnf,title):
    # calculate the score & draw the ROC curve
    for n in range(len(N)):
        trps = np.zeros((len(trpf),1))
        trns = np.zeros((len(trnf),1))
        alpha_sum = 0
        for i in range(N[n]):
            feature,theta,polarity,error,alpha = SC[i][:]
            if(polarity==1):
                trps[trpf[:,feature]<theta] += alpha
                trns[trnf[:,feature]<theta] += alpha
            else:
                trps[trpf[:,feature]>=theta] += alpha
                trns[trnf[:,feature]>=theta] += alpha
            alpha_sum += alpha
        trps = trps/alpha_sum
        trns = trns/alpha_sum
        roc_trpn = np.zeros((1000,1))
        roc_trnn = np.zeros((1000,1))
        
        for i in range(1000):
            roc_trpn[i] = np.sum(trps>=i/1000)/len(trpf)
            roc_trnn[i] = np.sum(trns>=i/1000)/len(trnf)
            
        # This is the ROC curve
        #plt.plot(roc_trpn,np.flip(roc_trnn,axis=1))
        plt.plot(roc_trnn, np.flip(roc_trpn, axis=1),label=N[n])
    plt.legend()
    plt.title(title)
    plt.show() 
#%%
ROC_curve(N,trpf,trnf,'train_ROC')  # train 
ROC_curve(N,tepf,tenf,'test_ROC')  # test

#%%
#==================================================
#     Q2 : get the faces of the picture  
#==================================================
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# open image
I = Image.open("D:/python/1809_智慧型/1127_pic.jpg").convert('L')
I = I.resize((np.array(I.size)*.1).astype(np.int32), Image.ANTIALIAS)
# put image into array
data = np.asarray(I)
plt.imshow(I)

#%%
# cut photos into lots of 19*19 
window = 19     # window size
w = I.size[0]   # 
h = I.size[1]
test = np.zeros(((h-window+1)*(w-window+1),window*window))
#for total in range(test.shape[0]):
i = 0
position = np.zeros(((h-window+1)*(w-window+1),2))
for y in range(h-window+1):
    for x in range(w-window+1):
        test[i] = data[y:y+window:1,x:x+window:1].flatten()
        position[i] = [x,y]
        i = i + 1

#%%
# feature extraction by ftable ： test 
testf = np.zeros((test.shape[0],fn))
for c in range(fn):
    testf[:,c] = fe(test,ftable,c)
#%%
# calculate the score of test 
tests = np.zeros((test.shape[0],1))
alpha_sum = 0
for i in range(50):
    feature,theta,polarity,error,alpha = SC[i][:]
    if(polarity==1):
        tests[testf[:,feature]<theta] += alpha
    else:
        tests[testf[:,feature]>=theta] += alpha
    alpha_sum += alpha
tests = tests/alpha_sum

print(np.sum(tests>=0.5)/trpn)

#%%
from PIL import Image as pimg

# Create figure and axes
fig,ax = plt.subplots(1)

I = Image.open("D:/python/1809_智慧型/1127_pic.jpg")
# Display the image
ax.imshow(I)
for k in range(tests.shape[0]):
    # Create a Rectangle patch
    if tests[k]>0.57:    
        rect = patches.Rectangle(position[k]*10,190,190,linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

plt.show()