
#Decision tree with random forest classifier
#to draw the graph we need python 2


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
from sklearn import tree

#for random forest

from sklearn.ensemble import RandomForestClassifier as RFC

#library to draw the graph

# from IPython.display import Image
# from sklearn.externals.six import StringIO
# import pydot


input_file="PastHires.csv" #input_file 
df=pd.read_csv(input_file,header=0)

#scikit-learn used everything as numerical value. So we are maping every non numerical
#to numerical values

d={'Y':1,'N':0}
df['Hired']=df['Hired'].map(d)
df['Employed?']=df['Employed?'].map(d)
df['Top-tier school']=df['Top-tier school'].map(d)
df['Interned']=df['Interned'].map(d)
d={'BS':0,'MS':1,'PhD':2}
df['Level of Education']=df['Level of Education'].map(d)

featurs=list(df.columns[0:6]) #taking the columns based on which we will take the decision

y=df['Hired']
x=df[featurs]

clf=tree.DecisionTreeClassifier()
clf.fit(x,y)


#code for showing the graph ##need python 2

# dot_data=StringIO()
# tree.export_graphviz(clf,out_file=dot_data,feature_names=featurs)
# graph=pydot.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())




#code for random forest

clf2=RFC(n_estimators=10)
clf2.fit(x,y)

print(clf.predict([[10,0,4,0,0,0]]))
print(clf2.predict([[10,0,4,0,0,0]]))

