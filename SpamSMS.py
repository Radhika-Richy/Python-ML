# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import nltk
nltk.download_shell() # download stopwords
#get the working directory and set the working directory
from os import chdir, getcwd
wd=getcwd()
wd
chdir(wd)

#rstrip() plus a list comprehension to get a list of all the lines of text messages
messages = [line.rstrip() for line in open('Desktop\SMSSpamCollection')]
print(len(messages))

#print the first ten messages and number them using enumerate
for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')
# data is a tab separated file import using pandas
import pandas as pd
messages = pd.read_csv('Desktop/SMSSpamCollection', sep='\t',
                           names=["label", "message"])
messages.head()
messages.describe()
messages.groupby('label').describe()

#make a new column to detect how long the text messages are
messages['length'] = messages['message'].apply(len)
messages.head()

import matplotlib.pyplot as plt
import seaborn as sns

messages['length'].plot(bins=50, kind='hist') 
messages.length.describe()
# print the longest message
messages[messages['length'] == 910]['message'].iloc[0]

#rying to see if message length is a distinguishing feature between ham and spam
messages.hist(column='length', by='label', bins=50,figsize=(12,4))

#create a function that will process the string in the message column,
#then we can just use apply() in pandas do process all the text in the DataFrame
#First removing punctuation.
import string
mess = 'Sample message! Notice: it has punctuation.'

# Check characters to see if they are in punctuation
nopunc = [char for char in mess if char not in string.punctuation]

# Join the characters again to form the string.
nopunc = ''.join(nopunc)
#impot a list of english stopwords from NLTK
from nltk.corpus import stopwords
stopwords.words('english')[0:10] # Show some stop words

nopunc.split()

# Now just remove any stopwords
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
clean_mess

#both of these together in a function to apply it to our DataFrame later on
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

messages['message'].head(5).apply(text_process)
messages.head()
#converting the normal text strings in to a list of tokens

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))

# take one text message and get its bag-of-words counts as a vector
message4 = messages['message'][3]
print(message4)

#vector representation of message4
bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)

#check and confirm which ones appear twice
print(bow_transformer.get_feature_names()[4068])
print(bow_transformer.get_feature_names()[9554])

messages_bow = bow_transformer.transform(messages['message'])

print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)

sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)

print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])
#transform the entire bag-of-words corpus into TF-IDF corpus at once
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', messages.label[3])

all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)

from sklearn.metrics import classification_report
print (classification_report(messages['label'], all_predictions))

from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
    
pipeline.fit(msg_train,label_train)
    
predictions = pipeline.predict(msg_test)
    
print(classification_report(predictions,label_test))  