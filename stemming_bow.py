# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os

CURR_DIR = os.getcwd()

# We load the database_dictionary created with the module preprocessing
pickle_in = open("database_dictionary.pickle", "rb")
database = pickle.load(pickle_in)
pickle_in.close()
print("Database 'database_dictionary.pickle' is loaded.")

# TODO : Compare with a dummy model who predict only the 4th score

X = []
y = []

dict_database = database.items()

for file in dict_database :
    # input : all the words of the file
    X.append([w for w in file[1][0]])

    # output of the TfidfVectorizer
    y.append(file[1][1])


# =============================================================================
# Dealing with unbalanced data ? Yes, maybe try to resample
# (sklearn.utils.resample or SMOTE)
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt

plt.figure()
pd.value_counts(y).plot.bar(title="SST score distribution")
plt.xlabel("SST score")
plt.ylabel("No. of text")
# =============================================================================
# Stemming process reduces the words to its’ root word.
# Unlike Lemmatization which uses grammar rules and dictionary
# for mapping words to root form, stemming simply removes suffixes/prefixes.
# =============================================================================

from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()

X = [[porter_stemmer.stem(w) for w in tokens] for tokens in X]

# =============================================================================
# Building dictionnary
# Each unique word would be identified by unique id in the dictionary object.
# =============================================================================

from gensim import corpora

mydict = corpora.Dictionary(X)
print("Total unique words: ", len(mydict.token2id))
print("\nSample data from dictionary:")
i = 0
# Print top 4 (word, id) tuples
for key in mydict.token2id.keys():
    print("Word: {} - ID: {} ".format(key, mydict.token2id[key]))
    if i == 3:
        break
    i += 1

# =============================================================================
# Generating BOW vectors
# Bag of Words (BOW) is one way of modeling text data for machine learning.
# This is the basic form of representing the text into numbers. Tokenized
# sentence is represented by an array of frequency of each word from the
# dictionary in the sentence.
# =============================================================================

import gensim
# =============================================================================
# print("Example of how BOW works")
# vocab_len = len(mydict)
# arr = []
# for line in X:
#     print("Doc2Bow Line:")
#     print(mydict.doc2bow(line))
#     for word in line:
#         arr.append(mydict.token2id[word])
#     print("Actual line:")
#     print(line)
#     print("(Word, count) Tuples:")
#     print([(mydict[id], count) for id, count in mydict.doc2bow(line) ])
#     print("Sparse bow vector for the line")
#     print(gensim.matutils.corpus2csc([mydict.doc2bow(line)],num_terms=vocab_len).toarray()[:,0])
#     break
# print("Sorted word id list")
# print(sorted(arr))
# =============================================================================


# Splitting into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, shuffle=True)

# saving the bow vector in a file
vocab_len = len(mydict)
bow_filename = 'train_review_bow.csv'
with open(bow_filename, 'w+') as bow_file:
    for line in X_train:
        features = gensim.matutils.corpus2csc([mydict.doc2bow(line)],num_terms=vocab_len).toarray()[:,0]
        line1 = ",".join( [str(vector_element) for vector_element in features] )
        bow_file.write(line1)
        bow_file.write('\n')

# =============================================================================
# Training with a decision tree classifier
# =============================================================================

from sklearn.tree import DecisionTreeClassifier
bow_df = pd.read_csv('train_review_bow.csv')
# Train the classifier with default parameters
# Initialize the classifier object
bow_clf = DecisionTreeClassifier(random_state=0)
# Fit the model with input vectors and corresponding sentiment labels
bow_clf.fit(bow_df, y_train[1:]) # je sais pas pourquoi mais il y a un label de trop
# il y a moyen que tout soit décalé de 1 ce qui fausserai le modèle

# =============================================================================
# Find out the most important features from the bow classification model
# _feature_importances_ attribute on the model can be used to get most important
#  features.
# =============================================================================
importances = list(bow_clf.feature_importances_)
feature_importances = [(feature, round(importance, 10)) for feature, importance in zip(bow_df.columns, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
top_i = 0
for pair in feature_importances:
    print('Variable: {:20} Importance: {}'.format(*pair))
    if top_i == 20:
        break
    top_i += 1

# Testing
from sklearn.metrics import classification_report, confusion_matrix
# Iterating through test data to get the predictions of sentiment by the model
test_features = []
for line in X_val:
    # Converting the tokens into the formet that the model requires
    features = gensim.matutils.corpus2csc([mydict.doc2bow(line)],num_terms=vocab_len).toarray()[:,0]
    test_features.append(features)
test_predictions = bow_clf.predict(test_features)
# Comparing the predictions to actual SST scores for the sentences
print("Précision, sensibilité, f-score\n", classification_report(y_val,test_predictions))
# accuracy : 0.38 tout pourri
print("Matrice de confusion\n", confusion_matrix(y_val, test_predictions))


# adapted from https://medium.com/swlh/sentiment-classification-with-bow-202c53dac154
