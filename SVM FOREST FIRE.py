# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:13:19 2023

@author: lenovo
"""



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score




df=pd.read_csv('forestfires.csv')
df



df.info()



df.shape



df.describe()



df.duplicated()



sns.pairplot(df)



sns.heatmap(df.isnull(),cmap='Reds')



sns.boxplot(data=df)



df1=df.iloc[:,2:]
df1



array = df1.values
X = array[:,0:28]
Y = array[:,28]




X



Y



X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3)



X_train.shape, y_train.shape, X_test.shape, y_test.shape


# #Grid Search CV

# ###rbf


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)



gsv.best_params_ , gsv.best_score_ 



clf = SVC(C= 15, gamma = 50)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# ###Linear


clf = SVC(kernel= "linear") 
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred) 



clf = SVC(kernel= "poly") 
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred) 
