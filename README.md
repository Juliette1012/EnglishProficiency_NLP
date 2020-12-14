# English proficiency prediction (NLP)

The aim of this project is to predict someone's English proficiency based on a text input.

Goal : Build a machine learning algorithm for predicting the SST score of each participant based on their transcript (we don't use the interviewer transcript).

Database : The NICT JLE Corpus available here : https://alaginrc.nict.go.jp/nict_jle/index_E.html

The source of the corpus data is the transcripts of the audio-recorded speech samples of 1,281 participants (1.2 million words, 300 hours in total) of English oral proficiency interview test. Each participant got a SST (Standard Speaking Test) score between 1 (low proficiency) and 9 (high proficiency) based on this test.

Our target is to beat the major class representation (class SST score 4) : 37%.

## Preprocessing (file preprocessing.py)

The preprocessing is made in extract_words() function :

* Using the HTMLParser class, we extract the participant transcript (all <B><B/> tags).
* We save the SST score of the participant with the SST_level tag.
* Inside participant transcript, we remove all other tags, punctuation and names using a regex. 
* Extract only English words and remove all stop words (a list of very common but useless for classification).
* Then comes a stemming or lemmatization phase (we started with a stemming phase and then moved to lemmatizer which gave us the best results).


## Process the data and prediction of the SST score

* Extract features with hesitation words (by hand) : hesitation.ipynb -> Bad results

* Extract features with the Bag of Words technique :
  * By hand : file handmade_bow.py
  * With gensim and DecisionTreeClassifier (sklearn) : file gensim_bow.py -> Bad results
  * With nltks library TfidfVectorizer and CountVectorizer : file nltk_vectorizer.ipynb

      * Only with score SST classes 4 5 6 et 7

            * Model SVC on tf_idf vectorizer
                * Accuracy :  0.5261627906976745

            * SVC on count vectorizer
                * Accuracy :  0.5668604651162791 (best score yet)

    * With all classes scores SST

            * SVC on tf_idf vectorizer
                * Accuracy :  0.4609375

            * SVC on count vectorizer
                * Accuracy :  0.4811111111111111


* Extract features with Glove technique : glove.ipynb
  * We create an embedding matrix of 100D using GloVe co-occurence probabilities on our unique words.
  * Using our embedding dictionary we create two neural networks to train (library keras)

    * Model 1 : LSTM + dense layer (activation sigmoid)
        * val_accuracy : 0.42

    * Model 2 : LSTM, globalMaxPool1D, Dropout (activation relu), Dense, Dropout (activtion relu)
        * val_accuracy : 0.48