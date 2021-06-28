import nltk
import pandas as pd
import csv
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
# import tensorflow_hub as hub
from tensorflow import keras
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Conv1D, MaxPool1D, MaxPooling1D
from keras.models import Model

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

file = open("Tweets.csv", "r", encoding='utf8')
reader = csv.reader(file)
poz = []
neg = []
neu = []
for d in reader:
    if d[1] == 'neutral':
        neu.append(d[10])
    if d[1] == 'positive':
        poz.append(d[10])
    if d[1] == 'negative':
        neg.append(d[10])

nltk.download('wordnet')
imdb = keras.datasets.imdb

word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}


def encode_review(text):
    lista = list(text.split(' '))
    ret = []
    for i in lista:
        if i.__contains__("@"):
            continue
        for j in word_index.keys():
            if i == j:
                ret.append(word_index.get(j))
    return ret


test_data_poz = test_data[:len(poz) - 2001]
test_labels_poz = test_labels[:len(poz) - 2001]
i = 0
while i < len(poz[2000:]):
    try:
        test_data_poz[i] = encode_review(poz[i])
        test_labels_poz[i] = 2
        i = i + 1
    except:
        i = i + 1

test_data_neg = test_data[7000:len(neg)-1]
test_labels_neg = test_labels[7000:len(neg)-1]
i = 0
while i < len(neg[7000:]):
    try:
        test_data_neg[i] = encode_review(neg[i])
        test_labels_neg[i] = 0
        i = i + 1
    except:
        i = i + 1
test_data_neu = test_data[2000:len(neu)-1]
test_labels_neu = test_labels[2000:len(neu)-1]
i = 0
while i < len(neu[2000:]):
    try:
        test_data_neu[i] = encode_review(neu[i])
        test_labels_neu[i] = 1
        i = i + 1
    except:
        i = i + 1

temp_data = np.concatenate((test_data_poz, test_data_neg))
temp_labels = np.concatenate((test_labels_poz, test_labels_neg))
test_data_done = np.concatenate((temp_data, test_data_neu))
test_labels_done = np.concatenate((temp_labels, test_labels_neu))
# test_data_done = test_data_poz + test_data_neg + test_data_neu
# test_labels_done = test_labels_poz + test_labels_neg + test_labels_neu


train_data_poz = train_data[:2000]
train_labels_poz = train_labels[:2000]
i = 0
while i < len(poz[:2000]):
    try:
        train_data_poz[i] = encode_review(poz[i])
        train_labels_poz[i] = 2
        i = i + 1
    except:
        i = i + 1
train_data_neg = train_data[2000:9000]
train_labels_neg = train_labels[2000:9000]
i = 0
while i < len(neg[:7000]):
    try:
        train_data_neg[i] = encode_review(neg[i])
        train_labels_neg[i] = 0
        i = i + 1
    except:
        i = i + 1
train_data_neu = train_data[9000:11000]
train_labels_neu = train_labels[9000:11000]
i = 0
while i < len(neu[:2000]):
    try:
        train_data_neu[i] = encode_review(neu[i])
        train_labels_neu[i] = 1
        i = i + 1
    except:
        i = i + 1


temp_data = np.concatenate((train_data_poz, train_data_neg))
temp_labels = np.concatenate((train_labels_poz, train_labels_neg))
train_data_done = np.concatenate((temp_data, train_data_neu))
train_labels_done = np.concatenate((temp_labels, train_labels_neu))
# train_data_done = train_data_poz + train_data_neg + train_data_neu
# train_labels_done = train_labels_poz + train_labels_neg + train_labels_neu


x_train = keras.preprocessing.sequence.pad_sequences(train_data_done,
                                                     padding='post',
                                                     maxlen=256)

y_train = keras.utils.to_categorical(train_labels_done, num_classes=3)

print(x_train)
print(y_train)
print(x_train.shape)
print(y_train.shape)
print(type(x_train))
print(type(y_train))

x_test = keras.preprocessing.sequence.pad_sequences(test_data_done,
                                                    padding='post',
                                                    maxlen=256)

y_test = keras.utils.to_categorical(test_labels_done, num_classes=3)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=256))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=200, batch_size=128)

score = model.evaluate(x_test, y_test, batch_size=128)

print(model.metrics_names)
print(score)
