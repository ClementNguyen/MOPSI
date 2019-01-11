import pandas as pd
import numpy as np
import os
import gensim 

os.chdir("C:\\Users\\quit7\\Desktop\\MOPSI\\code")

def createTrain(data):
    train = []
    for index,row in data.iterrows():
        train.append(row['product_name']+row['Description'])
    return train
    
def strToList(x):
    if type(x)==str:
        return x[2:-2].split("', '")
    
    

if __name__ == '__main__':
    
    model = gensim.models.Word2Vec(size=160, window=5, min_count=2, workers=10)
        
    for i in range(6):
        print("Chunk "+str(i)+"...")
        data = pd.read_csv(".\\data\\processed_data_"+str(i)+".csv",encoding = "ISO-8859-1")
        data = data[['product_name','Description']]
        data = data.dropna()
        data.product_name = data.product_name.apply(strToList)
        data.Description = data.Description.apply(strToList)
        
        train = createTrain(data)
        del data
        print("Training...")
        update_vocab = True
        if (i==0):
            update_vocab = False
        model.build_vocab(train,update=update_vocab)
        model.train(train,total_examples=len(train),epochs=10)
        
        
    model.save(".\\ml\\word2vec\\w2v_model")