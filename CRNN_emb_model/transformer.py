import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from eda import *
from sklearn.model_selection import train_test_split



class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
# (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
# print(len(x_train), "Training sequences")
# print(len(x_val), "Validation sequences")
#----------------------------------------------------------------------------

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
NB_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary

tk = Tokenizer(num_words=NB_WORDS,
               filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n',
               lower=True,
               split=" ")

tk.fit_on_texts(X_train)
X_train_seq = tk.texts_to_sequences(X_train)
X_test_seq = tk.texts_to_sequences(X_test)
MAX_LEN = 24

X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=maxlen)

le = LabelEncoder()
y_train_le = le.fit_transform(y_train)
y_test_le = le.transform(y_test)
y_train_oh = to_categorical(y_train_le)
y_test_oh = to_categorical(y_test_le)

X_train_emb, X_valid_emb, y_train_emb, y_valid_emb = train_test_split(X_train_seq_trunc, y_train_oh, test_size=0.1,random_state=37)

#----------------------------------------------------------------------------
# x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
# x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

# print(X_train_emb.shape)
# print(type(X_train_emb))
# print(X_train_emb)
# print('----------------')
# print(y_train_emb.shape)
# print(type(y_train_emb))
# print(y_train_emb)


embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    X_train_emb, y_train_emb, batch_size=32, epochs=2, validation_data=(X_valid_emb, y_valid_emb)
)
accr = model.evaluate(X_test_seq_trunc, y_test_oh, batch_size=128)
print(model.metrics)
print(accr)

