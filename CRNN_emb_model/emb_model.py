import pandas as pd
import numpy as np
import re
import collections
from pathlib import Path
from gensim import utils
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam
from eda import *

from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras import models
from keras import layers

NB_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary
VAL_SIZE = 1000  # Size of the validation set
NB_START_EPOCHS = 10  # Number of epochs we usually start to train with
BATCH_SIZE = 512  # Size of the batches used in the mini-batch gradient descent
MAX_LEN = 24  # Maximum number of words in a sequence
GLOVE_DIM = 100  # Number of dimensions of the GloVe word embeddings
root = Path('../')
input_path = root / 'input/'
ouput_path = root / 'output/'
source_path = root / 'source/'

early_stopping = EarlyStopping(monitor='loss', mode='min')


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


def test_model(model, X_train, y_train, X_test, y_test, epoch_stop):
    '''
    Function to test the model on new data after training it
    on the full training data with the optimal number of epochs.

    Parameters:
        model : trained model
        X_train : training features
        y_train : training target
        X_test : test features
        y_test : test target
        epochs : optimal number of epochs
    Output:
        test accuracy and test loss
    '''
    model.fit(X_train
              , y_train
              , epochs=epoch_stop
              , batch_size=BATCH_SIZE
              , verbose=0
              , callbacks=[early_stopping])
    results = model.evaluate(X_test, y_test)

    return results


def deep_model(model, X_train, y_train, X_valid, y_valid):
    '''
    Function to train a multi-class model. The number of epochs and
    batch_size are set by the constants at the top of the
    notebook.

    Parameters:
        model : model with the chosen architecture
        X_train : training features
        y_train : training target
        X_valid : validation features
        Y_valid : validation target
    Output:
        model training history
    '''
    model.compile(optimizer='rmsprop'
                  #  optimizer=SGD()
                  , loss='categorical_crossentropy'
                  , metrics=['accuracy'])

    history = model.fit(X_train
                        , y_train
                        , epochs=NB_START_EPOCHS
                        , batch_size=BATCH_SIZE
                        , validation_data=(X_valid, y_valid)
                        , verbose=0
                        , callbacks=[early_stopping])
    return history


def eval_metric(history, metric_name):
    '''
    Function to evaluate a trained model on a chosen metric.
    Training and validation metric are plotted in a
    line chart for each epoch.

    Parameters:
        history : model training history
        metric_name : loss or accuracy
    Output:
        line chart with epochs of x-axis and metric on
        y-axis
    '''
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]

    print(metric, val_metric)


df = pd.read_csv('Tweets.csv')
df = df.reindex(np.random.permutation(df.index))
num_of_negs = 0
length = len(df)
print(length)
i = 0
# while i < length:
#     try:
#         if df['airline_sentiment'][i] == 'negative':
#             if num_of_negs > 3000:
#                 df.drop(index=df[df['tweet_id'] == df['tweet_id'][i]].index, inplace=True)
#             num_of_negs = num_of_negs + 1
#         i = i + 1
#     except:
#         i = i + 1
print(len(df))
df = df[['text', 'airline_sentiment']]
df.text = df.text.apply(remove_stopwords).apply(remove_mentions)
# num_aug = 4
# alpha = 0.1
# length = len(df)
# i = 0
# j = 0
# while i < length:
#     if i % 1000 == 0:
#         print(i)
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
#         print('err')
#         i = i+1

X_train, X_test, y_train, y_test = train_test_split(df.text, df.airline_sentiment, test_size=0.1, random_state=37)
# EDA **********************************************************
# print(len(X_train), len(y_train))
# num_aug = 4
# alpha = 0.1
# length = len(X_train)
# i = 0
# j = 0
# for x, y in zip(X_train, y_train):
#     i = i + 1
#     try:
#         line = eda(x, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
#         while j < len(line):
#             # dataframe = pd.DataFrame([line[j], df['airline_sentiment'][i]], columns=list('test airline_sentiment'))
#             series1 = pd.Series(line[j], index=[length + i])
#             series2 = pd.Series(y, index=[length + i])
#             X_train = X_train.append(series1)
#             y_train = y_train.append(series2)
#             j = j+1
#         j = 0
#     except:
#         print('err')
#         continue
print(len(X_train), len(y_train))

tk = Tokenizer(num_words=NB_WORDS,
               filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n',
               lower=True,
               split=" ")

tk.fit_on_texts(X_train)
X_train_seq = tk.texts_to_sequences(X_train)
X_test_seq = tk.texts_to_sequences(X_test)

X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=MAX_LEN)

le = LabelEncoder()
y_train_le = le.fit_transform(y_train)
y_test_le = le.transform(y_test)
y_train_oh = to_categorical(y_train_le)
y_test_oh = to_categorical(y_test_le)

X_train_emb, X_valid_emb, y_train_emb, y_valid_emb = train_test_split(X_train_seq_trunc, y_train_oh, test_size=0.1,
                                                                      random_state=37)

emb_model = models.Sequential()
# emb_model.add(layers.Embedding(NB_WORDS, GLOVE_DIM, input_length=MAX_LEN))
# emb_model.add(layers.Flatten())
# emb_model.add(layers.Dense(3, activation='softmax'))

emb_model.add(layers.Embedding(NB_WORDS, 50, input_length=MAX_LEN))
emb_model.add(layers.Conv1DTranspose(64, 1, activation='sigmoid'))
emb_model.add(layers.MaxPooling1D(1, padding='same')) # ili 2
emb_model.add(layers.LSTM(64))
emb_model.add(layers.Dense(64, activation='relu'))
emb_model.add(layers.Dense(3, activation='softmax'))

# emb_model.add(layers.Embedding(NB_WORDS, 50, input_length=MAX_LEN))
# emb_model.add(layers.Conv1DTranspose(filters=64, kernel_size=1, padding='same', activation='sigmoid'))
# emb_model.add(layers.MaxPooling1D(pool_size=2))
# emb_model.add(layers.LSTM(80))
# # emb_model.add(layers.Bidirectional(layers.LSTM(64)))
# # emb_model.add(layers.Flatten())
# emb_model.add(layers.Dense(18, activation='relu'))
# emb_model.add(layers.Dropout(0.5))
# emb_model.add(layers.Dense(3, activation='softmax'))

# emb_model.add(layers.Embedding(NB_WORDS, GLOVE_DIM, input_length=MAX_LEN))
# emb_model.add(layers.Conv1D(64, 3, activation='relu'))
# emb_model.add(layers.MaxPooling1D(2, padding='same'))
# emb_model.add(layers.Bidirectional(layers.LSTM(64)))
# emb_model.add(layers.Dropout(0.5))
# emb_model.add(layers.Flatten())
# emb_model.add(layers.Dense(3, activation='softmax'))

emb_model.summary()
# emb_model.save('emb_model')
emb_history = deep_model(emb_model, X_train_emb, y_train_emb, X_valid_emb, y_valid_emb)
emb_results = test_model(emb_model, X_train_seq_trunc, y_train_oh, X_test_seq_trunc, y_test_oh, 10)
print('\n')
print('Test accuracy of word embeddings model: {0:.2f}%'.format(emb_results[1] * 100))
emb_model.save('emb_model')
var = emb_history.history['accuracy'][-1]
print(var)

eval_metric(emb_history, 'accuracy')
eval_metric(emb_history, 'loss')

# emb_results = test_model(emb_model, X_train_seq_trunc, y_train_oh, X_test_seq_trunc, y_test_oh, 6)
# print('\n')
# print('Test accuracy of word embeddings model: {0:.2f}%'.format(emb_results[1]*100))
