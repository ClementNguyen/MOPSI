import pandas as pd
import numpy as np
import os
import pickle
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from gensim.models import Word2Vec

os.chdir("C:\\Users\\quit7\\Desktop\\MOPSI\\code")
import text_processing

from similarity_train_set import products, products2


data = pd.read_csv("enpc_raw_data_products_ng.csv")
stopwords_ = set(stopwords.words('english'))


def evaluateSimilarity(index1,index2):
    w2v_model = Word2Vec.load(".\ml\word2vec\w2v_model")
    vocab = w2v_model.wv.vocab
    prediction_model = pickle.load(open(".\\ml\\similarity\\naive_bayes_w2v", 'rb'))
    index = [index1,index2]
    text1 = data.loc[index1,'Name']+data.loc[index1,'Description']
    text2 = data.loc[index2,'Name']+data.loc[index2,'Description']
    def processText(text):
        words = text_processing.tokenize(text)
        words = text_processing.normalize(words)
        new_words = []
        for word in words:
            if word not in stopwords_:
                new_words.append(word)
        return new_words
    text1 = processText(text1)
    text2 = processText(text2)
    
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
    def w2vScores(text1,text2):
        text1_afv = avg_feature_vector(text1)
        text2_afv = avg_feature_vector(text2)
        try:
            text_sim = max(0,1 - spatial.distance.cosine(text1_afv, text2_afv))
        except:
            text_sim = 0
        return [text_sim]
    
    x = [ w2vScores(text1,text2)+[abs(data.loc[index[0],'RetailPrice']-data.loc[index[1],'RetailPrice'])/data.loc[index[0],'RetailPrice']] ]
    
    return prediction_model.predict_proba(x)[0][1]
        