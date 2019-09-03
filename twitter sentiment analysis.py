# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:19:47 2019

@author: Vishal
"""

import re # for regular expressions
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import nltk # for text manipulation
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

df=pd.read_csv(r'C:\Users\Vishal\Desktop\rssfeedparser.csv')
df.head(2)
df.info()
#adding a class column, 1 is news and 0 is a article
df["class"]=0
df["class"][:70]=1
df["class"].value_counts()

length = df['title'].str.len()
#length = test['tweet'].str.len()

plt.hist(length, bins=10, label="length of title")
#plt.hist(length_test, bins=20, label="test_tweets")
plt.legend()
plt.show()

"""
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt   
"""
# remove twitter handles (@user) pick any word starting with @
#df['tidy_title'] = np.vectorize(remove_pattern)(df['tidy_title'], "@[\w]*")

# replace everything except characters and hashtags with spaces
df['tidy_title'] = df['title'].str.replace("[^a-zA-Z#]", " ")
#length 3 or les
df['tidy_title'] = df['tidy_title'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
df.head(2)

#Tokenization
tokenized_title = df['tidy_title'].apply(lambda x: x.split())
tokenized_title.head()

#Stemming
from nltk.stem.porter import *
stemmer = PorterStemmer()
tokenized_title = tokenized_title.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_title.head(2)

#combine tokens back together
for i in range(len(tokenized_title)):
    tokenized_title[i] = ' '.join(tokenized_title[i])

df['tidy_title'] = tokenized_title

all_words = ' '.join([text for text in df['tidy_title']])
# see which are the top most frequent words in the data
from nltk import FreqDist
# function to plot top n most frequent words
def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()  
  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
  
  # selecting top n most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()
freq_words(all_words)


negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])
freq_words(negative_words)
postive_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])
freq_words(postive_words)
#import wordcloud
#from wordcloud import WordCloud
#wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

#plt.figure(figsize=(10, 7))
#plt.imshow(wordcloud, interpolation="bilinear")
#plt.axis('off')
#plt.show()
# function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags
# extracting hashtags from non racist/sexist tweets

HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0])

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])

# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])

a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int) # calculating f1 score

test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('sub_lreg_bow.csv', index=False) # writing data to a CSV file

train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int)

