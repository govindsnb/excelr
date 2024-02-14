# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:41:47 2023

@author: lenovo
"""



import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
import seaborn as sn




book=pd.read_csv('book.csv')
book


# ### Apriori Algorithm

# If support value is 0.1 and threshold is 0.7.



frequent_itemsets = apriori(book, min_support=0.1, use_colnames=True)
frequent_itemsets




rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules
rules.sort_values('lift',ascending = False).head(10)
# (ItalCook)	(CookBks) are at the high confidence of 100%




rules.sort_values('lift',ascending = False)[0:20]



rules[rules.lift>1]




sn.scatterplot(x='support',y='confidence', data= rules)


# ###If suppose support value is 0.2 and threshold is 1.



frequent_itemsets = apriori(book, min_support=0.2, use_colnames=True)
frequent_itemsets




rules1 = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules1
rules1.sort_values('lift',ascending = False).head(10)
#(ChildBks)	(CookBks) are at the high confidence of 60.5%




rules1.sort_values('lift',ascending = False)[0:20]




rules1[rules1.lift>1]




sn.scatterplot(x='confidence',y='support',data=rules1)


# #Inference

# ###(ItalCook)	(CookBks) are the high confidence  of 100% at support value is 0.1 and threshold value is 0.7.
# 

# ###(ChildBks)	(CookBks) are at the high confidence of 60.5% at support value is 0.2 and threshold value is 1.




