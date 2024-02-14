# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 01:25:24 2023

@author: lenovo
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np




toyota=pd.read_csv('ToyotaCorolla.csv', encoding= 'unicode_escape') 
toyota





toyota.info()



toyota.isna().sum()




toyota.corr()



sns.set_style(style='darkgrid')
sns.pairplot(toyota)





model = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota).fit()





model.params





print(model.tvalues, '\n', model.pvalues)




(model.rsquared,model.rsquared_adj)



model.summary()




ml_cc=smf.ols('Price~cc',data = toyota).fit()  
print(ml_cc.tvalues, '\n', ml_cc.pvalues)  





ml_cc.summary()





ml_d=smf.ols('Price~Doors',data = toyota).fit()  
print(ml_d.tvalues, '\n', ml_d.pvalues) 




ml_d.summary()



qqplot=sm.qqplot(model.resid,line='q')
plt.title("Normal Q-Q plot of residuals")
plt.show()





list(np.where(model.resid>2100)) 




model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance





fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(toyota)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()





(np.argmax(c),np.max(c))




influence_plot(model)
plt.show()





k = toyota.shape[1]
n = toyota.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff





toyota[toyota.index.isin([80,221,960])]





toyota.head()





toyota_new=pd.read_csv('ToyotaCorolla.csv', encoding= 'unicode_escape')





toyota1=toyota_new.drop(toyota_new.index[[80,221,960]],axis=0).reset_index()
toyota1.shape





toyota2=toyota1.drop(['index'],axis=1)
toyota2.shape





final_ml_V= smf.ols('Price~Age_08_04+KM+HP+cc+Gears+Doors+Quarterly_Tax+Weight',data = toyota2).fit()





model_influence_V = final_ml_V.get_influence()
(c_V, _) = model_influence_V.cooks_distance





fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(toyota2)),np.round(c_V,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');




(np.argmax(c_V),np.max(c_V))





final_ml_V= smf.ols('Price~Age_08_04+KM+HP+cc+Gears+Doors+Quarterly_Tax+Weight',data =toyota2 ).fit()




(final_ml_V.rsquared,final_ml_V.aic)





new_data=pd.DataFrame({'Age_08_04':50,"KM":160,"HP":1100,"cc":225,"Gears":7,"Weight":250,"Doors":4,"Quarterly_Tax":350},index=[1])





final_ml_V.predict(new_data)
