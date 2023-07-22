# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 18:20:31 2023

@author: enesd
"""

import pandas as pd 
from scipy.cluster.hierarchy import linkage,dendrogram
import matplotlib.pyplot as plt 
import seaborn as sbn
from sklearn.cluster import AgglomerativeClustering

data=pd.read_csv("C:/Users/enesd/Downloads/Iris.csv")
dt=data.copy()

X=dt.drop(columns=["Id","Species"],axis=1)


hcsingle=linkage(X,method="single")
hccomplete=linkage(X,method="complete")
hcaverage=linkage(X,method="average")
hccentroid=linkage(X,method="centroid")
hcmedian=linkage(X,method="median")
hcward=linkage(X,method="ward")

fig,axes=plt.subplots(2,3)
dendrogram(hcsingle,ax=axes[0,0])
axes[0,0].set_title("Single")
dendrogram(hccomplete,ax=axes[0,1])
axes[0,1].set_title("Complete")
dendrogram(hcaverage,ax=axes[0,2])
axes[0,2].set_title("Average")
dendrogram(hccentroid,ax=axes[1,0])
axes[1,0].set_title("Centroid")
dendrogram(hcmedian,ax=axes[1,1])
axes[1,1].set_title("Median")
dendrogram(hcward,ax=axes[1,2])
axes[1,2].set_title("Ward")
plt.show()

model=AgglomerativeClustering(n_clusters=2,linkage="ward")
tahmin=model.fit_predict(X)

labels=model.labels_

sbn.scatterplot(x="SepalLengthCm",y="SepalWidthCm",data=X,hue=labels,palette="deep")
plt.show()












