# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 16:52:48 2023

@author: lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels import formula
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf





data = pd.read_csv('50_Startups.csv')
data





data.describe()





data.info()



#correlation
data.corr()





sns.pairplot(data)





sns.distplot(data['Profit'])




data = data.rename({'R&D Spend':'RD_spend','Marketing Spend':'Marketing_Spend'},axis=1)
data





data.drop('State',axis=1)





model = smf.ols("Profit~RD_spend+Administration+Marketing_Spend+Profit",data=data).fit()
model.summary()





model.params





print(model.tvalues, '\n', model.pvalues)





(model.rsquared,model.rsquared_adj)





md= smf.ols("Profit~RD_spend",data=data).fit()
print(md.tvalues, '\n' , md.pvalues)



md= smf.ols("Profit~Administration",data=data).fit()
print(md.tvalues, '\n' , md.pvalues)





md= smf.ols("Profit~RD_spend+Administration",data=data).fit()
md.summary()





rsq_RD = smf.ols("RD_spend~Marketing_Spend+Administration",data=data).fit().rsquared
vif_RD = 1/(1-rsq_RD) 
rsq_A = smf.ols("Administration~RD_spend+Marketing_Spend",data=data).fit().rsquared  
vif_A= 1/(1-rsq_A) 
rsq_M= smf.ols("Marketing_Spend~Administration+RD_spend",data=data).fit().rsquared  
vif_M = 1/(1-rsq_M) 
d1={'Variables':['Administration','RD_spend','Marketing_Spend'],'VIF':[vif_A,vif_RD,vif_M]}
vif_frame = pd.DataFrame(d1)
vif_frame





import statsmodels.api as sm
qqplot=sm.qqplot(model.resid,line='q') 
plt.title("Normal Q-Q plot of residuals")
plt.show()




def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()




plt.scatter(get_standardized_values(model.fittedvalues),
            get_standardized_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()




fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Administration", fig=fig)
plt.show()




fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "RD_spend", fig=fig)
plt.show()




fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Marketing_Spend", fig=fig)
plt.show()




model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance
c



fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(data)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()




(np.argmax(c),np.max(c))





from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()




k = data.shape[1]
n = data.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff



data[data.index.isin([47, 49])]




data_new=data.drop(data.index[[47,49]],axis=0).reset_index()





data_new=data_new.drop(['index'],axis=1)





data_new




final_Newdata= smf.ols('Profit~Administration+Marketing_Spend',data =data_new).fit()





(final_Newdata.rsquared,final_Newdata.aic)




final_Newdata= smf.ols('Profit~RD_spend+Marketing_Spend',data =data_new).fit()




(final_Newdata.rsquared,final_Newdata.aic)





new_data=pd.DataFrame({'Adiministration':100,'RD_spend':150,'Marketing_Spend':200},index=[1])
new_data




final_Newdata.predict(new_data)
