# coding=utf-8
from __future__ import print_function
import numpy as np
import codecs
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import *
from keras.layers import GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = []
with codecs.open('train/your training data.txt', 'r', 'utf-8') as fs:
    for line in fs:
        data.append(line.strip('\n').split('\t'))



wordlist = {}
num = 0
for text in data:
    if len(text)<3:
        print(text)
    else:
        for each in text[2]:
            if each!=' ':
                if each not in wordlist:
                    num += 1
                    wordlist[each] = num

with codecs.open('train/wordlist.txt', 'w', 'utf-8') as fds:
    for item in wordlist:
        fds.writelines([item+'\t'+str(wordlist[item])+'\n'])

print(len(wordlist))
sentence = []
label = []
for text in data:
    temp = []
    if text[0]!='' and text[0]!=' ':
        if text[0] == '负面':
            label.append([1,0,0])
        elif text[0] == '中性':
            label.append([0,1,0])
        elif text[0] == '正面':
            label.append([0,0,1])
        else:
            continue
        for each in text[2]:
            if each!=' ':
                temp.append(wordlist[each])
    sentence.append(temp)

X = np.array(sentence)
Y = np.array(label)

print(X.shape)
print(Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=123)

X_new = SelectKBest(chi2,k=5).fit_transform(x_train,y_train)
print(X_new.shape)


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.

    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))

def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

# Set parameters:
# ngram_range = 2 will add bi-grams features
ngram_range = 2
max_features = 20000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 100


if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in X:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))

# we add a GlobalAveragePooling1D, which will average the embeddings
# of all words in the document
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.2))
# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(3, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
# import pickle
# with open('sentiment_model_new.pickle', 'wb') as fs:
#     pickle.dump(model, fs)
with open('model/model_gram.json', 'w') as fout: fout.write(model.to_json())
with open('model/model_gram.yaml', 'w') as fout: fout.write(model.to_yaml())
print(model.predict(x_test,batch_size=batch_size,verbose=1))

