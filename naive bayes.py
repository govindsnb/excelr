# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:59:34 2023

@author: lenovo
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB




train=pd.read_csv('SalaryData_Train.csv')
train




test=pd.read_csv('SalaryData_Test.csv')
test




train.info()



test.info()




test.describe().round(2).style.background_gradient(cmap = 'Reds')




train.describe().round(2).style.background_gradient(cmap = 'Blues')




correlation = test.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')       
plt.title('Correlation between different fearures')




correlation = train.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')       
plt.title('Correlation between different fearures')




sns.heatmap(test.isnull(),cmap='Reds')




sns.heatmap(train.isnull(),cmap='Blues')



sns.pairplot(train)




sns.pairplot(test)




train[['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']] = train[['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']].apply(lambda x: pd.factorize(x)[0])
train




test[['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']] = test[['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']].apply(lambda x: pd.factorize(x)[0])
test




X=train.iloc[:,0:13]
Y=train.iloc[:,13]
x=test.iloc[:,0:13]
y=test.iloc[:,13]




X




Y




x




y


# #Naive Bayes

# ###Multinominal Naive Bayes



classifier_mb = MB()
classifier_mb.fit(X,Y)




train_pred_m = classifier_mb.predict(X)
accuracy_train_m = np.mean(train_pred_m==Y)




test_pred_m = classifier_mb.predict(x)
accuracy_test_m = np.mean(test_pred_m==y)




print('Training accuracy is:',accuracy_train_m,'\n','Testing accuracy is:',accuracy_test_m)


# ###Gaussian Naive Bayes



classifier_gb = GB()
classifier_gb.fit(X,Y) 




train_pred_g = classifier_gb.predict(X)
accuracy_train_g = np.mean(train_pred_g==Y)




test_pred_g = classifier_gb.predict(X)
accuracy_test_g = np.mean(test_pred_g==Y)




print('Training accuracy is:',accuracy_train_g,'\n','Testing accuracy is:',accuracy_test_g)
