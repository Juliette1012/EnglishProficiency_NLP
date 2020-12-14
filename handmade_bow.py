# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os

def createVector(database):
    vector = []
    dict_database = database.items()
    for file in dict_database : # browse all the files
        words = list(file[1][0]) #  because dictionnary = {file, [[all, words], level SST]}
        for word in words : # browse only all the words uses in the file
            if word not in vector :
                vector.append(word) # add a new word to the vector list
            else :
                pass
    return vector


def tokenize(database, vector):
    """
    This function enable to tokenize all the words in each file.
    """

    dict_database = database.items()
    features = []
    for file in dict_database :
        words = list(file[1][0])
        # Error : ValueError: maximum supported dimension for an ndarray is 32, found 15502
        vectorFile = np.zeros(len(vector)) # create a zeros-vector of the len of the vector word
        for w in vector : # to browse all the words in the general vector of words
            for word in words : # in the file
                if w == word :
                    vectorFile[np.where(vector == w)] = vectorFile[np.where(vector == w)] + 1 # add one more uses of this word in the file
        #print(vectorFile)
        features.append(vectorFile, file[1][1])
    return features



CURR_DIR = os.getcwd()
pickle_in = open("database_dictionary.pickle", "rb")
database = pickle.load(pickle_in)
pickle_in.close()

#len database : 1280 entries
#print(len(database))
vector = createVector(database)
#print(vector)

# len vector : 15502 words
#print(len(vector))
tokenize(database, vector)
