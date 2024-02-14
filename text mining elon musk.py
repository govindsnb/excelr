# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:55:49 2023

@author: lenovo
"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from nltk.corpus import stopwords
from textblob import TextBlob




data=pd.read_csv('Elon_musk.csv',encoding="latin-1")




data




data['word_count'] = data['Text'].apply(lambda x: len(str(x).split(" ")))
data[['Text','word_count']].head(10)




data['char_count'] = data['Text'].str.len()
data[['Text','char_count']].head(10)




def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))
data['avg_word'] = data['Text'].apply(lambda x: avg_word(x))
data[['Text','avg_word']].head(10)




import nltk
nltk.download('stopwords')
stop = stopwords.words('english')
data['stopwords'] = data['Text'].apply(lambda x: len([x for x in x.split() if x in stop]))
data[['Text','stopwords']].head(10)




data['hastags'] = data['Text'].apply(lambda x: len([x for x in x.split() if x.startswith('@')]))
data[['Text','hastags']].head(10)




data['numerics'] = data['Text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
data[['Text','numerics']].head(10)



data['upper'] = data['Text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
data[['Text','upper']].head(10)



data['Text'] = data['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data['Text'].head()



data['Text'] = data['Text'].str.replace('[^\w\s]','')
data['Text'].head()




stop = stopwords.words('english')
data['Text'] = data['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data['Text'].head()




freq = pd.Series(' '.join(data['Text']).split()).value_counts()[:10]
freq




freq = list(freq.index)
data['Text'] = data['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
data['Text'].head()




freq = pd.Series(' '.join(data['Text']).split()).value_counts()[-10:]
freq




freq = list(freq.index)
data['Text'] = data['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
data['Text'].head()




data['Text'][:5].apply(lambda x: str(TextBlob(x).correct()))



import nltk
nltk.download()




import nltk
nltk.download('punkt')
TextBlob(data['Text'][1]).words




from nltk.stem import PorterStemmer
st = PorterStemmer()
data['Text'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))



from textblob import Word
import nltk
nltk.download('wordnet')
data['Text'] = data['Text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['Text'].head()




TextBlob(data['Text'][0]).ngrams(2)




tf1 = (data['Text'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1




for i,word in enumerate(tf1['words']):
 tf1.loc[i, 'idf'] = np.log(data.shape[0]/(len(data[data['Text'].str.contains(word)])))
tf1




tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1




from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
stop_words= 'english',ngram_range=(1,1))
vect = tfidf.fit_transform(data['Text'])
vect


# ###Bag of words



from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
data_bow = bow.fit_transform(data['Text'])
data_bow


# ###Sentiment Analysis



data['Text'][:5].apply(lambda x: TextBlob(x).sentiment)




data['sentiment'] = data['Text'].apply(lambda x: TextBlob(x).sentiment[0] )
data[['Text','sentiment']].head(10)



from wordcloud import WordCloud, STOPWORDS
def plot_cloud(wordcloud):
    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud) 
    plt.axis("off");stopwords = STOPWORDS




data.plot.scatter(x='word_count',y='sentiment',figsize=(8,8),title='Sentence sentiment value to sentence word count')





