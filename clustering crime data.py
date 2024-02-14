# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:34:41 2023

@author: lenovo
"""




import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import warnings 
warnings.filterwarnings('ignore')





data=pd.read_csv('crime_data.csv')
data





data.info()




crime=data.drop("Unnamed: 0",axis=1)
crime


# Normalization



def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)




df_norm = norm_func(crime.iloc[:,:])
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




clf = KMeans(n_clusters=5)
y_kmeans = clf.fit_predict(df_norm)  




y_kmeans
#clf.cluster_centers_
clf.labels_ 




y_kmeans 




clf.cluster_centers_ 




clf.inertia_




md=pd.Series(y_kmeans)  # converting numpy array into pandas series object 
crime['clust']=md # creating a  new column and assigning it to new column 
crime




crime.groupby(crime.clust).mean() 




WCSS


# KMeans visualization



plt.figure(figsize=(15,8))
sn.scatterplot(crime['clust'],data['Unnamed: 0'],c=clf.labels_,s=300,marker='*')
plt.show();


# # DBSCAN Clustering



from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN




crime




array=crime.values
array




stscaler = StandardScaler().fit(array)
X = stscaler.transform(array) 
X  




dbscan = DBSCAN(eps=1.25, min_samples=5)
dbscan.fit(X)




#Noisy samples are given the label -1.
dbscan.labels_          




c=pd.DataFrame(dbscan.labels_,columns=['cluster'])   




c
pd.set_option("display.max_rows", None)  




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


# Silhoutte_score  



sklearn.metrics.silhouette_score(X, y_kmeans)


# DBSCAN Visualization



df.plot(x="Unnamed: 0",y ="cluster",c=dbscan.labels_ ,kind="scatter",s=50 ,cmap=plt.cm.copper_r) 
plt.title('Clusters using DBScan')      




plt.figure(figsize=(15,8))
sn.scatterplot(df1['Kcluster'],df1['Unnamed: 0'],c=clf.labels_,s=300,marker='*')
plt.show();




df1.plot(x="Unnamed: 0",y ="Kcluster",c=y_kmeans ,kind="scatter",s=50 ,cmap=plt.cm.copper_r) 
plt.title('Clusters using KMeans') 


# #HIERARCHAICAL Clustering



data




crime


# ####Standard Scaler



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
crime_subset = pd.DataFrame(scaler.fit_transform(crime.iloc[:,1:7]))
crime_subset  


# ###Dendrogram



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




p = np.array(df_norm) # converting into numpy array format 
z = linkage(df_norm, method="average",metric="euclidean")
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




p = np.array(df_norm) # converting into numpy array format 
z = linkage(df_norm, method="complete",metric="euclidean")
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




p = np.array(crime_subset) # converting into numpy array format 
z = linkage(crime_subset, method="complete",metric="euclidean")
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




from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=5, linkage='complete',affinity = "euclidean").fit(df_norm) 

cluster_labels=pd.Series(h_complete.labels_)
cluster_labels
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime   




crime.iloc[:,1:].groupby(crime.clust).mean()



data = crime[(crime.clust==0)]
data  




data = crime[(crime.clust==1)]
data  




data = crime[(crime.clust==2)]
data  




data = crime[(crime.clust==3)]
data  




data = crime[(crime.clust==4)]
data  


# ###Inference

# In Hierarchical cluster, Complete method is suitable for clustering the crime data.   