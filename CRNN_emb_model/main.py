import nltk
import pandas as pd
import csv
from keras.optimizers import SGD, Adam
from eda import *
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
# import tensorflow_hub as hub
from tensorflow import keras
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Conv1D, MaxPool1D, MaxPooling1D
from keras.models import Model

import imblearn as imb
imdb = keras.datasets.imdb

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
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


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


test_data_neg = []
test_labels_neg = []
i = 0
while i < len(neg[7000:]):
    try:
        test_data_neg.append(encode_review(neg[i]))
        test_labels_neg.append(0)
        i = i + 1
    except:
        i = i + 1
test_data_neg = np.array(test_data_neg, dtype=object)
test_labels_neg = np.array(test_labels_neg)
test_data_neu = []
test_labels_neu = []
i = 0
while i < len(neu[2000:]):
    try:
        test_data_neu.append(encode_review(neu[i]))
        test_labels_neu.append(1)
        i = i + 1
    except:
        i = i + 1
test_data_neu = np.array(test_data_neu, dtype=object)
test_labels_neu = np.array(test_labels_neu)
test_data_poz = []
test_labels_poz = []
i = 0
while i < len(poz[2000:]):
    try:
        test_data_poz.append(encode_review(poz[i]))
        test_labels_poz.append(2)
        i = i + 1
    except:
        i = i + 1
test_data_poz = np.array(test_data_poz, dtype=object)
test_labels_poz = np.array(test_labels_poz)

temp_data = np.concatenate((test_data_poz, test_data_neg))
temp_labels = np.concatenate((test_labels_poz, test_labels_neg))
test_data_done = np.concatenate((temp_data, test_data_neu))
test_labels_done = np.concatenate((temp_labels, test_labels_neu))

train_data_poz = []
train_labels_poz = []
i = 0
while i < len(poz[:2000]):
    try:
        train_data_poz.append(encode_review(poz[i]))
        train_labels_poz.append(2)
        i = i + 1
    except:
        i = i + 1
train_data_poz = np.array(train_data_poz, dtype=object)
train_labels_poz = np.array(train_labels_poz)
train_data_neg = []
train_labels_neg = []
i = 0
while i < len(neg[:7000]):
    try:
        train_data_neg.append(encode_review(neg[i]))
        train_labels_neg.append(0)
        i = i + 1
    except:
        i = i + 1
train_data_neg = np.array(train_data_neg, dtype=object)
train_labels_neg = np.array(train_labels_neg)
train_data_neu = []
train_labels_neu = []
i = 0
while i < len(neu[:2000]):
    try:
        train_data_neu.append(encode_review(neu[i]))
        train_labels_neu.append(1)
        i = i + 1
    except:
        i = i + 1
train_data_neu = np.array(train_data_neu, dtype=object)
train_labels_neu = np.array(train_labels_neu)

temp_data = np.concatenate((train_data_poz, train_data_neg))
temp_labels = np.concatenate((train_labels_poz, train_labels_neg))
train_data_done = np.concatenate((temp_data, train_data_neu))
train_labels_done = np.concatenate((temp_labels, train_labels_neu))

# deletes empty arrays ****************************************
i = 0
while i < len(train_data_done):
    if not train_data_done[i]:
        train_data_done = np.delete(train_data_done, i)
        train_labels_done = np.delete(train_labels_done, i)
    i = i + 1

i = 0
while i < len(test_data_done):
    if not test_data_done[i]:
        test_data_done = np.delete(test_data_done, i)
        test_labels_done = np.delete(test_labels_done, i)
    i = i + 1

# **************************************************************

# EDA **********************************************************
# num_aug = 4
# alpha = 0.1
# i = 0
# j = 0
# k = 0
# while i < len(train_data_done):
#     line = eda(decode_review(train_data_done[i]), alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
#     print(i)
#     while j < 5:
#         train_data_2[k] = encode_review(line[j])
#         train_labels_2[k] = train_labels_done[i]
#         j = j+1
#         k = k+1
#     i = i+1
#     k = k+1
#     j = 0


x_train = keras.preprocessing.sequence.pad_sequences(train_data_done,  # train_data_done
                                                     padding='post',
                                                     maxlen=256)
x_test = keras.preprocessing.sequence.pad_sequences(test_data_done,
                                                    padding='post',
                                                    maxlen=256)

y_train = keras.utils.to_categorical(train_labels_done, num_classes=3)  # train_labels_done
y_test = keras.utils.to_categorical(test_labels_done, num_classes=3)

max_words = 100000
max_len = 256


def crnn():
    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(max_words, 32, input_length=max_len)(inputs)
    layer = Conv1D(32, 1, activation='relu')(layer)  # bilo 256
    layer = MaxPooling1D(2, padding='same')(layer)
    layer = LSTM(16)(layer)  # 64
    layer = Dense(16, name='FC1')(layer)  # poslednje1# 64
    layer = Activation('relu')(layer)  # poslednje1
    # layer = Dense(8, name='FC2')(layer)  # poslednje1# 64
    # layer = Activation('relu')(layer)  # poslednje1
    #
    layer = Dropout(0.1)(layer)  # poslednje2

    layer = Dense(3, name='out_layer')(layer)
    layer = Activation('softmax')(layer)

    model = Model(inputs=inputs, outputs=layer)

    return model


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model = crnn()
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]
# train_data_done,
# #train_labels_done,

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=10,
                    batch_size=128,  # 512 l:0.723, a:0.401 256 l:0.712 a:0.393
                    validation_data=(x_val, y_val),
                    # validation_split=0.1,
                    verbose=1)

accr = model.evaluate(x_test, y_test, batch_size=128)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

print(model.metrics_names)
print(accr)
