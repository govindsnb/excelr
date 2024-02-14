# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:12:32 2023

@author: lenovo
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve



bank=pd.read_csv('bank-full.csv',sep =';' )
bank.head()



bank.info()




bank.shape




bank[categorical].isnull().sum()


# ###Factorization

# 



bank[['job','marital','education','default','housing','loan','contact','month','poutcome','y']]=bank[['job','marital','education','default','housing','loan','contact','month','poutcome','y']].apply(lambda x: pd.factorize(x)[0])
bank               #converting into dummy variables




X = bank.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
Y = bank.iloc[:,16]
classifier = LogisticRegression()
classifier.fit(X,Y) 



classifier.coef_  # coefficients of features   



classifier.predict_proba (X) # Probability values   


# # Prediction



y_pred = classifier.predict(X)
bank["y_pred"] = y_pred
bank




y_prob = pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))
new_df = pd.concat([bank,y_prob],axis=1)
new_df  


# ##confusion matrix


confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix) 



pd.crosstab(y_pred,Y)  



#type(y_pred)
accuracy = sum(Y==y_pred)/bank.shape[0]
accuracy



print (classification_report (Y, y_pred))  



Logit_roc_score=roc_auc_score(Y,classifier.predict(X))
Logit_roc_score                                   # logistic ROC score 


# ###ROC_Curve


fpr, tpr, thresholds = roc_curve(Y,classifier.predict_proba(X)[:,1]) 
plt.plot(fpr, tpr, label='Logistic Regression (area=%0.2f)'% Logit_roc_score)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])                 
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')    
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()                                                      #fpr, tpr, thresholds = precision-recall_curve(Y,classifier.predict_proba(X)[:,1]) 



y_prob1 = pd.DataFrame(classifier.predict_proba(X)[:,1]) 
y_prob1                              












