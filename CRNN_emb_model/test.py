import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import pandas as pd
import numpy as np
import re
import collections
from pathlib import Path
from gensim import utils
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam

from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras import models
from keras import layers
from eda import  *
# x_train = np.random.random((1000, 20))
# y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
# x_test = np.random.random((100, 20))
# y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
#
# print(x_train)
# print(y_train)
# print(x_train.shape)
# print(y_train.shape)
# print(type(x_train))
# print(type(y_train))
#
# print(x_train[1])
#
#
# model = Sequential()
# model.add(Dense(64, activation='relu', input_dim=20))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))
#
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=100, batch_size=128)
#
# score = model.evaluate(x_test, y_test, batch_size=128)
#
# print(model.metrics_names)
# print(score)

NB_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary
VAL_SIZE = 1000  # Size of the validation set
NB_START_EPOCHS = 10  # Number of epochs we usually start to train with
BATCH_SIZE = 512  # Size of the batches used in the mini-batch gradient descent
MAX_LEN = 24  # Maximum number of words in a sequence
def remove_stopwords(input_text):
    '''
    Function to remove English stopwords from a Pandas Series.

    Parameters:
        input_text : text to clean
    Output:
        cleaned Pandas Series
    '''
    stopwords_list = stopwords.words('english')
    # Some words which might indicate a certain sentiment are kept via a whitelist
    whitelist = ["n't", "not", "no"]
    words = input_text.split()
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]
    return " ".join(clean_words)


def remove_mentions(input_text):
    '''
    Function to remove mentions, preceded by @, in a Pandas Series

    Parameters:
        input_text : text to clean
    Output:
        cleaned Pandas Series
    '''
    return re.sub(r'@\w+', '', input_text)

df = pd.read_csv('Tweets.csv')
df = df.reindex(np.random.permutation(df.index))
df = df[['text', 'airline_sentiment']]
df.text = df.text.apply(remove_stopwords).apply(remove_mentions)


# EDA **********************************************************
# num_aug = 4
# alpha = 0.1
# length = len(df)
# i = 0
# j = 0
# while i < length:
#     try:
#         line = eda(df['text'][i], alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
#         while j < len(line):
#             # dataframe = pd.DataFrame([line[j], df['airline_sentiment'][i]], columns=list('test airline_sentiment'))
#             dataframe = pd.DataFrame({'text': line[j],
#                                       'airline_sentiment': df['airline_sentiment'][i]},
#                                      index=[length+i+1])
#             df.append(dataframe)
#             j = j+1
#         i = i+1
#         j = 0
#     except:
#         i = i + 1
# print(df)


X_train, X_test, y_train, y_test = train_test_split(df.text, df.airline_sentiment, test_size=0.1, random_state=37)
print(len(X_train))
print(len(y_train))
# EDA **********************************************************
num_aug = 4
alpha = 0.1
length = len(X_train)
i = 0
j = 0
for x, y in zip(X_train, y_train):
    i = i + 1
    try:
        line = eda(x, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        while j < len(line):
            # dataframe = pd.DataFrame([line[j], df['airline_sentiment'][i]], columns=list('test airline_sentiment'))
            series1 = pd.Series(line[j], index=[length + i])
            series2 = pd.Series(y, index=[length + i])
            X_train = X_train.append(series1)
            y_train = y_train.append(series2)
            j = j+1
        j = 0
    except:
        print('err')
        continue

print(len(X_train))
print(len(y_train))
# tk = Tokenizer(num_words=NB_WORDS,
#                filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n',
#                lower=True,
#                split=" ")
#
# tk.fit_on_texts(X_train)
# X_train_seq = tk.texts_to_sequences(X_train)
# X_test_seq = tk.texts_to_sequences(X_test)
#
# X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=MAX_LEN)
# X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=MAX_LEN)
#
# le = LabelEncoder()
# y_train_le = le.fit_transform(y_train)
# y_test_le = le.transform(y_test)
# y_train_oh = to_categorical(y_train_le)
# y_test_oh = to_categorical(y_test_le)
#
# X_train_emb, X_valid_emb, y_train_emb, y_valid_emb = train_test_split(X_train_seq_trunc, y_train_oh, test_size=0.1,
#                                                                       random_state=37)
#
