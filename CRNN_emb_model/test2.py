import pandas as pd
import numpy as np
import re
import collections
from pathlib import Path
from gensim import utils
from tensorflow import keras
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
df = df[['text', 'airline_sentiment']]
df.text = df.text.apply(remove_stopwords).apply(remove_mentions)
X_train, X_test, y_train, y_test = train_test_split(df.text, df.airline_sentiment, test_size=0.1, random_state=37)

print(y_train)
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
print(y_train_le)
y_test_le = le.transform(y_test)
y_train_oh = to_categorical(y_train_le)
print(y_train_oh)
y_test_oh = to_categorical(y_test_le)

X_train_emb, X_valid_emb, y_train_emb, y_valid_emb = train_test_split(X_train_seq_trunc, y_train_oh, test_size=0.1,
                                                                      random_state=37)

model = keras.models.load_model('emb_model')
prediction_ch = model.predict(X_train_seq_trunc)
print(prediction_ch)
print(y_train)



# PRETVORENO U FUNKCIJU get_prediction() ***************************************************
# prepare_country_hall = pd.Series()
# i = 0
# for review in review_country_hall:
#     series = pd.Series(review, index=[i])
#     prepare_country_hall = prepare_country_hall.append(series)
#     i = i + 1
#
# tk.fit_on_texts(prepare_country_hall)
# prepare_ch = tk.texts_to_sequences(prepare_country_hall)
# prepared_ch = pad_sequences(prepare_ch, maxlen=MAX_LEN)
#
# prediction_ch = model.predict(prepared_ch)
# predicted_ch = []
# for pred in prediction_ch:
#     check = [pred[0], pred[1], pred[2]]
#     max_value = max(check)
#     max_index = check.index(max_value)
#     if max_index == 1:
#         predicted_ch.append(0)  # negative
#     if max_index == 2:
#         predicted_ch.append(-1)  # neutral
#     if max_index == 3:
#         predicted_ch.append(1)  # positive
# ******************************************************************************************

# PRETVORENO U FUNKCIJU get_values_from_csv() ***************************************************
# review_country_hall = []
# sentiment_country_hall = []
# for line in country_hall_csv:
#     review_country_hall.append(line[5])
#     if line[9] == 1:
#         sentiment_country_hall.append(1)
#     else:
#         sentiment_country_hall.append(0)
#
# review_leeds = []
# sentiment_leeds = []
# for line in leeds_csv:
#     review_leeds.append(line[5])
#     if line[9] == 1:
#         sentiment_leeds.append(1)
#     else:
#         sentiment_leeds.append(0)
#
# review_park_royal = []
# sentiment_park_royal = []
# for line in park_royal_csv:
#     review_park_royal.append(line[5])
#     if line[9] == 1:
#         sentiment_park_royal.append(1)
#     else:
#         sentiment_park_royal.append(0)
#
# review_riverbank = []
# sentiment_riverbank = []
# for line in riverbank_csv:
#     review_riverbank.append(line[5])
#     if line[9] == 1:
#         sentiment_riverbank.append(1)
#     else:
#         sentiment_riverbank.append(0)
#
# review_victoria_london = []
# sentiment_victoria_london = []
# for line in victoria_london_csv:
#     review_victoria_london.append(line[5])
#     if line[9] == 1:
#         sentiment_victoria_london.append(1)
#     else:
#         sentiment_victoria_london.append(0)
#
# review_victoria = []
# sentiment_victoria = []
# for line in victoria_csv:
#     review_victoria.append(line[5])
#     if line[9] == 1:
#         sentiment_victoria.append(1)
#     else:
#         sentiment_victoria.append(0)
#
# review_vondelpark = []
# sentiment_vondelpark = []
# for line in vondelpark_csv:
#     review_vondelpark.append(line[5])
#     if line[9] == 1:
#         sentiment_vondelpark.append(1)
#     else:
#         sentiment_vondelpark.append(0)
#
# review_westminster = []
# sentiment_westminster = []
# for line in westminster_csv:
#     review_westminster.append(line[5])
#     if line[9] == 1:
#         sentiment_westminster.append(1)
#     else:
#         sentiment_westminster.append(0)
#
# review_wroclaw = []
# sentiment_wroclaw = []
# for line in wroclaw_csv:
#     review_wroclaw.append(line[5])
#     if line[9] == 1:
#         sentiment_wroclaw.append(1)
#     else:
#         sentiment_wroclaw.append(0)
#
# review_dubai = []
# sentiment_dubai = []
# for line in dubai_csv:
#     review_dubai.append(line[5])
#     if line[9] == 1:
#         sentiment_dubai.append(1)
#     else:
#         sentiment_dubai.append(0)
#
# review_edinburgh = []
# sentiment_edinburgh = []
# for line in edinburgh_csv:
#     review_edinburgh.append(line[5])
#     if line[9] == 1:
#         sentiment_edinburgh.append(1)
#     else:
#         sentiment_edinburgh.append(0)
#
# review_edwardian = []
# sentiment_edwardian = []
# for line in edwardian_csv:
#     review_edwardian.append(line[5])
#     if line[9] == 1:
#         sentiment_edwardian.append(1)
#     else:
#         sentiment_edwardian.append(0)
#
# review_edwardian_london = []
# sentiment_edwardian_london = []
# for line in edwardian_london_csv:
#     review_edwardian_london.append(line[5])
#     if line[9] == 1:
#         sentiment_edwardian_london.append(1)
#     else:
#         sentiment_edwardian_london.append(0)
#
# review_glasgow = []
# sentiment_glasgow = []
# for line in glasgow_csv:
#     review_glasgow.append(line[5])
#     if line[9] == 1:
#         sentiment_glasgow.append(1)
#     else:
#         sentiment_glasgow.append(0)
#
# review_liverpool = []
# sentiment_liverpool = []
# for line in liverpool_csv:
#     review_liverpool.append(line[5])
#     if line[9] == 1:
#         sentiment_liverpool.append(1)
#     else:
#         sentiment_liverpool.append(0)
#
# review_manchester = []
# sentiment_manchester = []
# for line in manchester_csv:
#     review_manchester.append(line[5])
#     if line[9] == 1:
#         sentiment_manchester.append(1)
#     else:
#         sentiment_manchester.append(0)
#
# review_sydney = []
# sentiment_sydney = []
# for line in sydney_csv:
#     review_sydney.append(line[5])
#     if line[9] == 1:
#         sentiment_sydney.append(1)
#     else:
#         sentiment_sydney.append(0)
# ******************************************************************************************
