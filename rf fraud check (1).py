#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[2]:


df = pd.read_csv('Fraud_check.csv')


# In[3]:


df


# In[4]:


df=pd.get_dummies(df,columns=['Undergrad','Marital.Status','Urban'], drop_first=True)
df["TaxInc"] = pd.cut(df["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])
df
df = pd.get_dummies(df,columns = ["TaxInc"],drop_first=True)


# In[5]:


import seaborn as sns
sns.pairplot(data=df, hue = 'TaxInc_Good')
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
df_norm = norm_func(df.iloc[:,1:])
df_norm.tail(10)
X = df_norm.drop(['TaxInc_Good'], axis=1)
y = df_norm['TaxInc_Good']


# In[6]:


from sklearn.model_selection import train_test_split
# Splitting data into train & test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)
##Converting the Taxable income variable to bucketing. 
df_norm["income"]="<=30000"
df_norm.loc[df["Taxable.Income"]>=30000,"income"]="Good"
df_norm.loc[df["Taxable.Income"]<=30000,"income"]="Risky"
df.drop(["Taxable.Income"],axis=1,inplace=True)
df.rename(columns={"Undergrad":"undergrad","Marital.Status":"marital","City.Population":"population","Work.Experience":"experience","Urban":"urban"},inplace=True)


# In[7]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass
features = df.iloc[:,0:5]
labels = df.iloc[:,5]


# In[8]:


colnames = list(df.columns)
predictors = colnames[0:5]
target = colnames[5]


# In[9]:


##Splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)


# In[17]:


from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)
model.estimators_
model.classes_
model.n_classes_
model.oob_score_
prediction = model.predict(x_train)


# In[18]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)
pred_test = model.predict(x_test)
acc_test =accuracy_score(y_test,pred_test)


# In[19]:


acc_test


# In[ ]:




