# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 00:47:01 2023

@author: lenovo
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')
glass = pd.read_csv('glass.csv')
glass
glass['Type'].value_counts()
glass.info()
glass[glass.duplicated()].shape
glass[glass.duplicated()]

df = glass.drop_duplicates()
df
corr = df.corr()
corr
sns.heatmap(corr)
sns.scatterplot(df['RI'],df['Na'],hue=df['Type'])
sns.pairplot(df,hue='Type')
plt.show()
df
DF= df.iloc[:,0:9]
DF

array= DF.values
array
from sklearn.preprocessing import StandardScaler
# Normalization function
stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)
X
df_knn = pd.DataFrame(X,columns=df.columns[:-1])
df_knn
x= df_knn
y= df['Type']
x
y
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.3,random_state=45)
x_train

x_test
y_train

y_test
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
KNeighborsClassifier(n_neighbors=3)
#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category

pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions

print("Accuracy", accuracy_score(y_test,preds)*100)
Accuracy 65.625
model.score(x_train,y_train)
0.825503355704698
print(classification_report(y_test,preds))
