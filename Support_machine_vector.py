#with fake generated data

import random
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
import statsmodels.api as sm
from mpl_toolkits import mplot3d
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

from sklearn import svm,datasets

def creatClusterData(N,k):
    random.seed(5) #to get same data everytime
    pointsPerCluster=float(N)/k
    X=[]
    y=[]
    for i in range(k):
        incomeCentroid=random.uniform(20000.0,200000.0)
        ageCentroid=random.uniform(20.0,70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid,10000.0),np.random.normal(ageCentroid,2.0)])
            y.append(i)
    X=np.array(X)
    y=np.array(y)
    return X,y

def plotPrediction(X,clf):
    xx,yy=np.meshgrid(np.arange(0,250000,10),np.arange(10,70,0.5))
    z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    plt.figure(figsize=(10,7))
    z=z.reshape(xx.shape)
    plt.contourf(xx,yy,z,cmap=plt.cm.Paired,alpha=0.8)
    plt.scatter(X[:,0],X[:,1],c=y.astype(np.float))
    plt.show()

X,y=creatClusterData(100,5)

plt.figure(figsize=(10,7))
plt.scatter(X[:,0],X[:,1],c=y.astype(np.float))
plt.show()

svc=svm.SVC(kernel='linear',C=1.0).fit(X,y)

plotPrediction(X,svc)
