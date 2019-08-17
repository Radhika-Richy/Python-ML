# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 23:06:03 2019

@author: Vishal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import data
yelp = pd.read_csv('Desktop/yelp.csv')
yelp.head(0)
yelp.info()

#Creating a new column "text length", number of words in the text column.
yelp['text length'] = yelp['text'].apply(len)

#Use FacetGrid from the seaborn library to create histograms of text length based on the star ratings
g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length',bins=50)

#boxplot
sns.boxplot(x='stars',y='text length', data=yelp)
#countplot of the number of occurrences for each type of star rating
sns.countplot(x='stars',data=yelp)


yelp.corr()
# create a heatmap based on .corr()
sns.heatmap(yelp.corr(),cmap='coolwarm',annot=True)

#X will be the 'text' column of yelp_class and y will be the 'stars' column
X = yelp['text']
y = yelp['stars']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)
predictions = nb.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


#try to include TF-IDF to this process using a pipeline
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
X = yelp['text']
y = yelp['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
