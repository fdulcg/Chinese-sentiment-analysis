import codecs
import jieba
import numpy as np
import pickle
from jieba import analyse
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.feature_selection import VarianceThreshold

data = []
with codecs.open('your-training-data.txt', 'r', 'utf-8') as fs:
    for line in fs:
        data.append(line.strip('\n').split('\t'))
neg = []
with codecs.open('negative.txt', 'r', 'utf-8') as fs:
    for line in fs:
        neg.append(line.strip('\n').strip('\r'))
neg = set(neg)
pos = []
with codecs.open('positive.txt', 'r', 'utf-8') as fs:
    for line in fs:
        pos.append(line.strip('\n').strip('\r'))
pos = set(pos)

# text = [elem[3] for elem in data[1:]]
# txt = ' '.join(text)
# content = ' '.join(text)

sent_tags = []
for text in data[0:]:
    if len(text)<3:
        print(text)
    else:
        nn = jieba.cut(text[2])
        sent_tags.append(' '.join([elem for elem in nn if elem in pos or elem in neg]))

content = ' '.join(sent_tags)
top_tags = 450
tags = analyse.extract_tags(content, topK=top_tags)
with open('../model/sentiment.pickle', 'wb') as fs:
    pickle.dump(tags, fs)

feature_vec = []
for text in data[0:]:
    nn = jieba.cut(text[2])
    nn = set(list(nn))
    f_vec = np.zeros(top_tags).astype(int)
    for ind, elem in enumerate(tags):
        if elem in nn:
            f_vec[ind] = 1
    feature_vec.append([text[0]] + f_vec.tolist())
with open('feature_part.csv', 'w') as fs:
    head = ['label']
    for _ in range(top_tags):
        head.append('f' + str(_))
    fs.write('\t'.join(head) + '\n')

with open('feature_part.csv', 'a+') as fs:
    for elem in feature_vec:
        # if elem[0] == '负面': label = 'neg'
        # if elem[0] == '正面': label = 'pos'
        # if elem[0] == '中性': label = 'cent'
        out_str = '\t'.join(map(str, elem)) + '\n'
        fs.write(out_str)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
predata = []
with open('feature_part.csv') as fs:
    for line in fs:
        predata.append(line.strip('\n').split('\t'))
# label, feature = zip(*[[elem[0], elem[1:]] for elem in predata[1:]])
label = []
feature = []
pos = 0
neg = 0
mid = 0
for elem in predata[1:]:
    feature.append(elem[1:])
    label.append(elem[0])
    if elem[0]=='正面':
        pos += 1
    elif elem[0]=='中性':
        mid += 1
    else:
        neg += 1
print("正面：%d 中性 %d  负面%d   "%(pos,mid,neg))

import pandas
from pandas import Series,DataFrame


featurenp = np.array(feature)
labelnp = np.array(label)
X_train, X_test, Y_train, Y_test = train_test_split(featurenp, labelnp, test_size=0.1, random_state=0)

# sel = VarianceThreshold()
# X_train = sel.fit_transform(X_train)
print(X_train.shape)
k_value = 12
# for k_value in range(3,450):
if True:
    X_new = DataFrame(data=X_train)
    Y_new = DataFrame(data=Y_train)

    X_testN = DataFrame(data=X_test)
    Y_testN = DataFrame(data=Y_test)
    X_testN = SelectKBest(chi2, k=k_value).fit_transform(X_testN, Y_testN)

    X_new = SelectKBest(chi2,k=k_value).fit_transform(X_new,Y_new)
    print(X_new.shape)

# print(label)
# print(feature)
# rf = RandomForestClassifier(n_estimators=100)
# model = rf.fit(X=X_train, y=Y_train)

    rf = RandomForestClassifier(n_estimators=100,oob_score=True,max_depth=20)
# rf = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo')
    model = rf.fit(X=X_new,y=Y_train)
# model = rf.fit(X=X_train,y=Y_train)
    value = model.score(X_testN,Y_test)
    # if value>maxNum:
    #     maxNum = value
    #     maxv = k_value
    print(str(value))


# print(model.score(X_test,Y_test))

with open('../model/sentimentpickle', 'wb') as fs:
    pickle.dump(rf, fs)

