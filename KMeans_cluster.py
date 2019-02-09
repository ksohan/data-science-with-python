
#KMeans is implemented on fake data 

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

def creatClusterData(N,k):
    random.seed(10) #to get same data everytime
    pointsPerCluster=float(N)/k
    X=[]
    for i in range(k):
        incomeCentroid=random.uniform(20000.0,200000.0)
        ageCentroid=random.uniform(20.0,70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid,10000.0),np.random.normal(ageCentroid,2.0)])
    X=np.array(X)
    return X
    

data=creatClusterData(1000,5)
model=KMeans(n_clusters=5)
model=model.fit(scale(data)) #scaling for good fit

print(model.labels_) #looking at the clusters points

plt.figure(figsize=(8,6))
plt.scatter(data[:,0],data[:,1],c=model.labels_.astype(np.float)) #giving color based on labels
plt.show()





