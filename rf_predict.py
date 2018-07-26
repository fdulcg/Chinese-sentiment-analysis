#-*-coding:utf-8-*-
import pickle
import jieba
import numpy as np
from snownlp import SnowNLP
import pandas
from pandas import DataFrame
tags = pickle.load(open('model/sentiment_tags.pickle','rb'))
rf_model = pickle.load(open('model/sentiment_model.pickle','rb'))

def Run(input_sent, sentiment_tags=tags, model=rf_model):
    nn = jieba.cut(input_sent)
    nn = set(list(nn))
    showlist = []
    f_vec = np.zeros(len(sentiment_tags)).astype(int)
    for ind, word in enumerate(sentiment_tags):
        if word in nn :
            showlist.append(word)
            f_vec[ind] = 1
    # print(str(f_vec))

    X_new = DataFrame(data=f_vec)
    label = model.predict_proba(f_vec.reshape(1,-1))[0]
    labelclass = model.predict(f_vec.reshape(1,-1))[0]
    return labelclass

def Sentiment(sent):
    label = Run(sent, tags, rf_model)
    return label
