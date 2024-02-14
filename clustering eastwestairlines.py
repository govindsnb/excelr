# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:38:37 2023

@author: lenovo
"""



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering



data=pd.read_excel('EastWestAirlines.xlsx',sheet_name='data')
data.head()



data.info()




data.shape



air=data.drop(['ID#','Award?'], axis=1)
air


# ####Normalization 



def norm_func(i):
  x = (i-i.min())/(i.max()-i.min())
  return (x)



df_norm = norm_func(air.iloc[:,:])
df_norm 


# #KMEANS Clustering



fig = plt.figure(figsize=(10, 8))
WCSS = []
for i in range(1, 11):
    clf = KMeans(n_clusters=i)
    clf.fit(df_norm)
    WCSS.append(clf.inertia_) # inertia is another name for WCSS
plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show()  




WCSS




clf = KMeans(n_clusters=4)
y_kmeans = clf.fit_predict(df_norm)  




y_kmeans




clf.cluster_centers_ 




clf.inertia_




md=pd.Series(y_kmeans)  # converting numpy array into pandas series object 
air['clust']=md # creating a  new column and assigning it to new column 
air




air.groupby(air.clust).mean() 




plt.figure(figsize=(15,8))
sn.scatterplot(air['clust'],data['ID#'],c=clf.labels_,s=300,marker='*')
plt.show();


# #DBSCAN Clustering



from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler




air




array=air.values
array




stscaler = StandardScaler().fit(array)
X = stscaler.transform(array) 
X  




dbscan = DBSCAN(eps=0.70, min_samples=10)
dbscan.fit(X)




dbscan.labels_ 




c=pd.DataFrame(dbscan.labels_,columns=['cluster'])  




c




df = pd.concat([data,c],axis=1)  
df   




d1=dbscan.labels_
d1



import sklearn
sklearn.metrics.silhouette_score(X, d1)




from sklearn.cluster import KMeans
clf = KMeans(n_clusters=5)
y_kmeans = clf.fit_predict(X)




y_kmeans




cl1=pd.DataFrame(y_kmeans,columns=['Kcluster']) 
cl1




df1 = pd.concat([df,cl1],axis=1) 
df1 


# silhouette_score



sklearn.metrics.silhouette_score(X, y_kmeans)


# DBSCAN Visualization



df.plot(x="ID#",y ="cluster",c=dbscan.labels_ ,kind="scatter",s=50 ,cmap=plt.cm.copper_r) 
plt.title('Clusters using DBScan')
plt.xlabel("ID#")
plt.ylabel("cluster")




df1.plot(x="ID#",y ="Kcluster",c=y_kmeans ,kind="scatter",s=50 ,cmap=plt.cm.copper_r) 
plt.title('Clusters using KMeans') 




plt.figure(figsize=(20,10))
sn.scatterplot(df1['Kcluster'],df1['ID#'],c=clf.labels_,s=400,marker='*')
plt.show();


# # HIERARCHAICAL Clustering



data




air=data.drop(['ID#','Award?'],axis=1)




air




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
air_subset = pd.DataFrame(scaler.fit_transform(air.iloc[:,1:7]))
air_subset


# Dendrogrom



from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 
p = np.array(df_norm) # converting into numpy array format 
z = linkage(df_norm, method="single",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(
    z,
    #leaf_rotation=6.,  # rotates the x axis labels
    #leaf_font_size=15.,  # font size for the x axis labels
)
plt.show()  




p = np.array(df_norm) 
z = linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(
    z,
)
plt.show()    




p = np.array(df_norm) 
z = linkage(df_norm, method="average",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(
    z,
    
)
plt.show()    




from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=5, linkage='complete',affinity = "euclidean").fit(df_norm) 

cluster_labels=pd.Series(h_complete.labels_)
cluster_labels
air['clust']=cluster_labels   
air




air.iloc[:,1:].groupby(air.clust).mean()




data = air[(air.clust==0)]
data  



data = air[(air.clust==1)]
data  




data = air[(air.clust==2)]
data  




data = air[(air.clust==3)]
data  




data = air[(air.clust==4)]
data  


# #Inference
# 

# In Hierarichical clustering, complete method is suitable to form cluster for EastWestairlines.