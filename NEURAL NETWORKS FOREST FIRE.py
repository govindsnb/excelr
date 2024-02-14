# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:10:58 2023

@author: lenovo
"""



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()



df = pd.read_csv("forestfires.csv")



df1 = df.copy()
df1



df1 = df1.drop(["month","day"], axis = 1)
df1.info()



df.info()



df1.isnull().sum()



df1.describe()



df1["size_category"] = le.fit_transform(df1["size_category"])



df1.size_category.value_counts()



# outlier check:


ax = sns.boxplot(df1['area'])



# There are 3 Outlier instances in our data.



plt.rcParams["figure.figsize"] = 9,4
plt.figure(figsize=(16,5))
print("Skew: {}".format(df1['area'].skew()))
print("Kurtosis: {}".format(df1['area'].kurtosis()))
ax = sns.kdeplot(df1['area'],shade=True,color='g')
plt.xticks([i for i in range(0,1200,50)])
plt.show()



# The Data is highly skewed and has large kurtosis value. Majority of the forest fires do not cover a large area, most of the damaged area is under 100 hectares of land



dfa = df1[df1.columns[0:10]]
month_colum = dfa.select_dtypes(include='object').columns.tolist()
num_columns = dfa.select_dtypes(exclude='object').columns.tolist()
plt.figure(figsize=(18,40))
for i,col in enumerate(num_columns,1):
    plt.subplot(8,4,i)
    sns.kdeplot(df[col],color='g',shade=True)
    plt.subplot(8,4,i+10)
    df[col].plot.box()
plt.tight_layout() 
plt.show()
num_data = df[num_columns]
pd.DataFrame(data=[num_data.skew(),num_data.kurtosis()],index=['skewness','kurtosis'])



sns.distplot(df1.size_category)




x = df1.iloc[:,0:28]
y = df1.iloc[:,28]



x



y



x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.3, stratify = y)
scaled_values = StandardScaler()
scaled_values.fit(x_train)



x_train = scaled_values.transform(x_train)
x_test = scaled_values.transform(x_test)




from sklearn.neural_network import MLPClassifier



mlp = MLPClassifier(hidden_layer_sizes = (10,10))



y_train = y_train.astype(int)
x_test = x_test.astype(int)




mlp.fit(np.array(x_train), np.array(y_train))



prediction_train = mlp.predict(x_train)
prediction_test = mlp.predict(x_test)
prediction_test



type(prediction_test)



y_test



pd.Series(prediction_test)



type(y_test)



from sklearn.metrics import classification_report, confusion_matrix
np.mean(y_test == prediction_test)
np.mean(y_train == prediction_train)




