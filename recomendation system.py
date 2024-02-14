# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:46:08 2023

@author: lenovo
"""

import pandas as pd
import numpy as np
import matplotlib as plt




book=pd.read_csv('book.csv',encoding='latin-1')
book




df =book.drop(['Unnamed: 0'],axis=1)




df =df.rename({'User.ID':'user_id','Book.Title':'book_title','Book.Rating':'book_rating'},axis=1)




df.info()




len(df.user_id.unique())




len(df.book_title.unique())




df1 = df.drop_duplicates(['user_id','book_title'])




books = df1.pivot(index='user_id',
                                 columns='book_title',
                                 values='book_rating').reset_index(drop=True)




books




books.index = df.user_id.unique()




books




books.fillna(0, inplace=True)




books




from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation




df2 = 1 - pairwise_distances( books.values,metric='cosine')




df2




#Store the results in a dataframe
books2 = pd.DataFrame(df2)




books2.index = df1.user_id.unique()
books2.columns = df1.user_id.unique()




books2.iloc[0:5, 0:5]




np.fill_diagonal(df2, 0)
books2.iloc[0:5, 0:5]




books2.idxmax(axis=1)[0:5]
