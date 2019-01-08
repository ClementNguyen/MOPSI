import pandas as pd
import numpy as np
import os
import sys
import pickle
import itertools
import nltk 
from sklearn import model_selection, metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords

os.chdir("C:\\Users\\quit7\\Desktop\\MOPSI\\code")
import parse 
import text_processing

from similarity_train_set import products, products2


## Similarity prediction by counting words occurences

def train_similarity():
    """
    Train a model of prediction of the similarity between two products based on MultinomialNB method.
    The model take in parameters the number of words that appears in both descriptions/name and the price difference.
    """
    data = pd.read_csv('.\\ml\\similarity\\products_sample.csv')
    train_set = pd.read_csv('.\\ml\\similarity\\similarity_training_set.csv')
    
    def strToListInt(x):
        """
        Transform '[1,2]' into [1,2]
        """
        if type(x)==str:
            return [int(i) for i in x[1:-1].split(", ")]
    train_set.Index =train_set.Index.apply(strToListInt)
    
    def strToList(x):
        """
        Transform '['a','b']' into ['a','b']
        """
        if type(x)==str:
            return x[2:-2].split("', '")
    data.Name = data.Name.apply(strToList)
    data.Description = data.Description.apply(strToList)
    
    #model = LogisticRegression(solver = 'lbfgs')
    model = MultinomialNB()
    x_train = []
    y_train = train_set['Similarity'].values
    
    def countSimilarWords(index):
        count = 0
        name1 = set(data.loc[index[0],'Name'])
        desc1 = set(data.loc[index[0],'Description'])
        name2 = set(data.loc[index[1],'Name'])
        desc2 = set(data.loc[index[1],'Description'])
        for x in name1:
            if (x in name2):
                count += 1
        for x in desc1:
            if (x in desc2):
                count += 1
        return count/(len(name1)+len(name2)+len(desc1)+len(desc2))
    
    print('Preparing training set...')
    for i in range(len(train_set)):
        if (i%50000==0):
            print(i)
        index = train_set.loc[i,'Index']
        x_train += [ [ countSimilarWords(index), abs(data.loc[index[0],'RetailPrice']-data.loc[index[1],'RetailPrice'])/data.loc[index[0],'RetailPrice'] ] ]
        #x_train += [ [ abs(data.loc[index[0],'RetailPrice']-data.loc[index[1],'RetailPrice'])/data.loc[index[0],'RetailPrice'] ] ]
        
    x_train,x_test,y_train,y_test = model_selection.train_test_split(x_train,y_train,test_size=0.2)
    
    #Train model
    print('Training model...')
    model.fit(x_train,y_train)
    #Save model
    pickle.dump(model, open('.\\ml\\similarity\\naive_bayes', 'wb'))
    
    #Evaluate accuracy
    x_test_similar = []
    y_test_similar = []
    x_test_unsimilar = []
    y_test_unsimilar = []
    for i in range(len(y_test)):
        if (y_test[i]==1):
            x_test_similar += [x_test[i]]
            y_test_similar += [y_test[i]]
        else :
            x_test_unsimilar += [x_test[i]]
            y_test_unsimilar += [y_test[i]]
    predictions = model.predict(x_test)
    print('Overall accuracy : ',metrics.accuracy_score(predictions,y_test))
    predictions = model.predict(x_test_similar)
    print('Accuracy (for similar products) : ',metrics.accuracy_score(predictions,y_test_similar))
    predictions = model.predict(x_test_unsimilar)
    print('Accuracy (for unsimilar products) : ',metrics.accuracy_score(predictions,y_test_unsimilar))
    
    # Print: 
    # Overall accuracy :  0.720595453461792
    # Accuracy (for similar products) :  0.8133531406788688
    # Accuracy (for unsimilar products) :  0.6277154083637384

        



## Similarity prediction using Word2Vec

from gensim.models import Word2Vec
from scipy import spatial #for cosine

def train_similarity_w2v():
    
    data = pd.read_csv('.\\ml\\similarity\\products_sample.csv')
    train_set = pd.read_csv('.\\ml\\similarity\\similarity_training_set.csv')
    
    def strToList(x):
        """
        Transform '[1,2]' into [1,2]
        """
        if type(x)==str:
            return [int(i) for i in x[1:-1].split(", ")]
    train_set.Index =train_set.Index.apply(strToList)
    
    def strToList(x):
        """
        Transform '['a','b']' into ['a','b']
        """
        if type(x)==str:
            return x[2:-2].split("', '")
    data.Name = data.Name.apply(strToList)
    data.Description = data.Description.apply(strToList)
    
    #Prediction model
    model = MultinomialNB()
    #Word2Vec model
    w2v_model = Word2Vec.load(".\\ml\\word2vec\\w2v_model")
    vocab = w2v_model.wv.vocab
    
    x_train = []
    y_train = train_set['Similarity'].values
    
    stopwords_ = set(stopwords.words('english'))
    def remove_stopwords(words):
        """
        Remove stopwords
        """
        new_words = []
        for word in words:
            if word not in stopwords_:
                new_words.append(word)
        return new_words
        
    index2word_set = set(w2v_model.wv.index2word)
    def avg_feature_vector(words):
        num_features = 160
        feature_vec = np.zeros((num_features, ), dtype='float32')
        n_words = 0
        for word in words:
            if word in index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, w2v_model[word])
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec
    
    def w2vScores(index):
        name1 = remove_stopwords(data.loc[index[0],'Name'])
        desc1 = remove_stopwords(data.loc[index[0],'Description'])
        name2 = remove_stopwords(data.loc[index[1],'Name'])
        desc2 = remove_stopwords(data.loc[index[1],'Description'])
        text1 = name1 + desc1
        text2 = name2 + desc2
        text1_afv = avg_feature_vector(text1)
        text2_afv = avg_feature_vector(text2)
        try:
            text_sim = max(0,1 - spatial.distance.cosine(text1_afv, text2_afv))
        except:
            text_sim = 0
        return [text_sim]
        
        #https://stackoverflow.com/questions/22129943/how-to-calculate-the-sentence-similarity-using-word2vec-model-of-gensim-with-pyt
    
    print('Preparing training set...')
    for i in range(len(train_set)):
        if (i%10000==0):
            print(i)
        index = train_set.loc[i,'Index']
        x_train += [ w2vScores(index)+[abs(data.loc[index[0],'RetailPrice']-data.loc[index[1],'RetailPrice'])/data.loc[index[0],'RetailPrice']] ]
        
    x_train,x_test,y_train,y_test = model_selection.train_test_split(x_train,y_train,test_size=0.2)
    
    #Train model
    print('Training model...')
    model.fit(x_train,y_train)
    #Save model
    pickle.dump(model, open('.\\ml\\similarity\\naive_bayes_w2v', 'wb'))
    
    #Evaluate accuracy
    x_test_similar = []
    y_test_similar = []
    x_test_unsimilar = []
    y_test_unsimilar = []
    for i in range(len(y_test)):
        if (y_test[i]==1):
            x_test_similar += [x_test[i]]
            y_test_similar += [y_test[i]]
        else :
            x_test_unsimilar += [x_test[i]]
            y_test_unsimilar += [y_test[i]]
    predictions = model.predict(x_test)
    print('Overall accuracy : ',metrics.accuracy_score(predictions,y_test))
    predictions = model.predict(x_test_similar)
    print('Accuracy (for similar products) : ',metrics.accuracy_score(predictions,y_test_similar))
    predictions = model.predict(x_test_unsimilar)
    print('Accuracy (for unsimilar products) : ',metrics.accuracy_score(predictions,y_test_unsimilar))

    # Print:
    # Overall accuracy :  0.7331958994208142
    # Accuracy (for similar products) :  0.9049878567517864
    # Accuracy (for unsimilar products) :  0.5617601322480772







