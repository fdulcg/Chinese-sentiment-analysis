# coding=utf-8
from keras.models import model_from_yaml , load_model
from keras.preprocessing import sequence
import numpy as np
import codecs

word2vec = {line.strip('\n').split('\t')[0]: line.strip('\n').split('\t')[1] for line in codecs.open('train/wordlist.txt','r','utf-8')}
sent2label = {line2.strip('\n').split('\t')[2]: line2.strip('\n').split('\t')[0] for line2 in codecs.open('train/your training data.txt','r','utf-8')}
X_train = []
labeltest = []
for key in sent2label:
    labeltest.append(sent2label[key])
    temp = []
    for each in key:
        if each in word2vec.keys():
            temp.append(word2vec[each])
    X_train.append(temp)
X_train = np.array(X_train)
X_train = sequence.pad_sequences(X_train, maxlen=350)
print(X_train.shape)
# with open('model/model_gram.yaml') as fin: model = model_from_yaml(fin.read())
model = load_model('model/model.h5')
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Z = model.predict(X_train, batch_size=16)

score = 0
for e in range(len(Z)):
    prolist = Z[e]
    max = 0
    pos = 1000
    for i in range(3):
        if prolist[i]>max:
            max = prolist[i]
            pos = i
    des = ''
    print(str(prolist)+'  '+str(pos)+' '+labeltest[e])
    if pos==0:
        des = '负面'
    elif pos==1:
        des = '中性'
    elif pos == 2:
        des = '正面'
    else:
        pass
    if des == labeltest[e]:
        score +=1
    else:
        pass

print(score)
