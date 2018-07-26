import codecs
import jieba
import numpy as np
import pickle
from jieba import analyse

data = []
with codecs.open('0904-sentiment.csv', 'r', 'utf-8') as fs:
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
for text in data[1:]:
    nn = jieba.cut(text[3])
    sent_tags.append(' '.join([elem for elem in nn if elem in pos or elem in neg]))

content = ' '.join(sent_tags)
top_tags = 500
tags = analyse.extract_tags(content, topK=top_tags)
# with open('sentiment_tags_svm.pickle', 'wb') as fs:
#     pickle.dump(tags, fs)

feature_vec = []
for text in data[1:]:
    nn = jieba.cut(text[3])
    nn = set(list(nn))
    f_vec = np.zeros(top_tags).astype(int)
    for ind, elem in enumerate(tags):
        if elem in nn:
            f_vec[ind] = 1
    feature_vec.append([text[1]] + f_vec.tolist())
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
from sklearn import svm
from sklearn.model_selection import train_test_split

predata = []
with open('feature_part.csv') as fs:
    for line in fs:
        predata.append(line.strip('\n').split('\t'))

v
label, feature = zip(*[[elem[0], elem[1:]] for elem in predata[1:]])
X_train, X_test, Y_train, Y_test = train_test_split(featurenp, labelnp, test_size=0.2, random_state=0)

# print(label)
# print(feature)
rf = RandomForestClassifier(n_estimators=100)
model = rf.fit(X=X_train, y=Y_train)


