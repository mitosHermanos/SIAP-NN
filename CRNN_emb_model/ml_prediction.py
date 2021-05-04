import numpy as np
import pandas as pd
import re
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, naive_bayes, svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

MAX_LEN = 24  # Maximum number of words in a sequence
NB_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary


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
while i < length:
    try:
        if df['airline_sentiment'][i] == 'negative':
            if num_of_negs > 3000:
                df.drop(index=df[df['tweet_id'] == df['tweet_id'][i]].index, inplace=True)
            num_of_negs = num_of_negs + 1
        i = i + 1
    except:
        i = i + 1
print(len(df))
df = df[['text', 'airline_sentiment']]
df.text = df.text.apply(remove_stopwords).apply(remove_mentions)

X_train = df.text
y_train = df.airline_sentiment
tk = Tokenizer(num_words=NB_WORDS,
               filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n',
               lower=True,
               split=" ")

tk.fit_on_texts(X_train)
X_train_seq = tk.texts_to_sequences(X_train)

X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=MAX_LEN)

le = LabelEncoder()
y_train_le = le.fit_transform(y_train)
y_train_oh = to_categorical(y_train_le)

random_forest = RandomForestClassifier()
random_forest.fit(X_train_seq_trunc, y_train_oh)

count_vect_nb = CountVectorizer()
X_train_counts_nb = count_vect_nb.fit_transform(X_train)
tfidf_transformer_nb = TfidfTransformer()
X_train_tfidf_nb = tfidf_transformer_nb.fit_transform(X_train_counts_nb)
nb = naive_bayes.MultinomialNB()
nb.fit(X_train_tfidf_nb, y_train)

count_vect_svm = CountVectorizer()
X_train_counts_svm = count_vect_svm.fit_transform(X_train)
tfidf_transformer_svm = TfidfTransformer()
X_train_tfidf_svm = tfidf_transformer_svm.fit_transform(X_train_counts_svm)
svm = LinearSVC()
svm.fit(X_train_tfidf_svm, y_train)


def get_predictions_svm(reviews):
    prepare1 = pd.Series(dtype=object)
    i = 0
    for review in reviews:
        series = pd.Series(review, index=[i])
        prepare1 = prepare1.append(series)
        i = i + 1

    X_train_counts = count_vect_svm.transform(prepare1)

    prediction = svm.predict(X_train_counts)
    predicted = []
    for pred in prediction:

        check = [pred[0], pred[1], pred[2]]
        max_value = max(check)
        max_index = check.index(max_value)
        if max_index == 0:
            predicted.append(0)  # negative
        if max_index == 1:
            predicted.append(-1)  # neutral
        if max_index == 2:
            predicted.append(1)  # positive
    return predicted


def get_predictions_nb(reviews):
    prepare1 = pd.Series(dtype=object)
    i = 0
    for review in reviews:
        series = pd.Series(review, index=[i])
        prepare1 = prepare1.append(series)
        i = i + 1

    X_train_counts = count_vect_nb.transform(prepare1)

    prediction = nb.predict(X_train_counts)
    predicted = []
    for pred in prediction:

        check = [pred[0], pred[1], pred[2]]
        max_value = max(check)
        max_index = check.index(max_value)
        if max_index == 0:
            predicted.append(0)  # negative
        if max_index == 1:
            predicted.append(-1)  # neutral
        if max_index == 2:
            predicted.append(1)  # positive
    return predicted


def get_predictions_rf(reviews):
    prepare1 = pd.Series(dtype=object)
    i = 0
    for review in reviews:
        series = pd.Series(review, index=[i])
        prepare1 = prepare1.append(series)
        i = i + 1

    tk.fit_on_texts(prepare1)
    prepare2 = tk.texts_to_sequences(prepare1)
    prepared = pad_sequences(prepare2, maxlen=MAX_LEN)

    prediction = random_forest.predict(prepared)
    predicted = []
    for pred in prediction:
        # neg = pred[0]
        # poz = pred[1] + pred[2]
        # if neg > poz:
        #     predicted.append(0)
        # else:
        #     predicted.append(1)
        check = [pred[0], pred[1], pred[2]]
        max_value = max(check)
        max_index = check.index(max_value)
        if max_index == 0:
            predicted.append(0)  # negative
        if max_index == 1:
            predicted.append(-1)  # neutral
        if max_index == 2:
            predicted.append(1)  # positive
    return predicted


all_hotels_predict_row = open("../all_hotels_predict_row.csv", "r")
all_hotels_predict_categorized = open("../all_hotels_predict_categorized.csv", "r")

all_hotels_predict_row_csv = csv.reader(all_hotels_predict_row)
all_hotels_predict_categorized_csv = csv.reader(all_hotels_predict_categorized)

def get_values_from_csv(csv):
    review = []
    sentiment = []
    for line in csv:
        review.append(line[7])
        if line[10] == 1:
            sentiment.append(1)
        else:
            sentiment.append(0)
    print('get_values_from_csv', review[1])
    return review, sentiment

ahpr = get_values_from_csv(all_hotels_predict_row_csv)[0]
ahpc = get_values_from_csv(all_hotels_predict_categorized_csv)[0]

ahpr_rf = get_predictions_rf(ahpr)
ahpr_nb = get_predictions_nb(ahpr)
ahpr_svm = get_predictions_svm(ahpr)
ahpc_rf = get_predictions_rf(ahpc)
ahpc_nb = get_predictions_nb(ahpc)
ahpc_svm = get_predictions_svm(ahpc)

def add_column_to_csv_row(rf, nb, svm, csv1, name_of_csv):
    list0 = []
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []
    list8 = []
    list9 = []
    list10 = []
    list11 = []
    list12 = []
    list13 = []
    list14 = []
    list15 = []
    list16 = []
    list17 = []
    list18 = []
    list19 = []
    list20 = []
    list21 = []
    list22 = []
    list23 = []
    list24 = []
    list25 = []
    list26 = []
    list27 = []
    list28 = []
    list29 = []
    list30 = []
    corr = 0
    for line, r, n, s in zip(csv1, rf, nb, svm):
        list0.append(line[0])
        list1.append(line[1])
        list2.append(line[2])
        list3.append(line[3])
        list4.append(line[4])
        list5.append(line[5])
        list6.append(line[6])
        list7.append(line[7])
        list8.append(line[8])
        list9.append(line[9])
        list10.append(line[10])
        list11.append(line[11])

        list15.append(line[12])
        list16.append(line[13])
        list17.append(line[14])
        list18.append(line[15])
        list19.append(line[16])
        list20.append(line[17])
        list21.append(line[18])
        list22.append(line[19])
        list23.append(line[20])
        list24.append(line[21])
        list25.append(line[22])
        list26.append(line[23])
        list27.append(line[24])
        list28.append(line[25])
        list29.append(line[26])
        list30.append(line[27])

        if r == 0:
            i = 0
        elif r == 1:
            i = 1
        else:
            if line[10] == '1':
                i = 1
            else:
                i = 0
        list12.append(i)

        if n == 0:
            i = 0
        elif n == 1:
            i = 1
        else:
            if line[10] == '1':
                i = 1
            else:
                i = 0
        list13.append(i)

        if s == 0:
            i = 0
        elif s == 1:
            i = 1
        else:
            if line[10] == '1':
                i = 1
            else:
                i = 0
        list14.append(i)
    print('asdasdasd',list0[1])
    name_dict = {
        'Hotel_name': list0,
        'Hotel_chain': list1,
        'Guest_country': list2,
        'Room_info': list3,
        'Nights_stayed': list4,
        'Date of stay': list5,
        'Travel_type': list6,
        'Review': list7,
        'Grade': list8,
        'Title': list9,
        'Positive': list10,
        'CRNN': list11,
        'Random Forest': list12,
        'Naive Bayes': list13,
        'SVM': list14,
        'staff': list15,
        'location': list16,
        'food/drink': list17,
        'dirty': list18,
        'bed': list19,
        'comfort': list10,
        'price': list21,
        'bathroom': list22,
        'parking': list23,
        'restaurant': list24,
        'noisiness': list25,
        'tv': list26,
        'internet': list27,
        'fitness': list28,
        'covid': list29,
        'temperature': list30
    }
    df = pd.DataFrame(name_dict)
    df.to_csv(name_of_csv, index=False)


def add_column_to_csv_cat(rf, nb, svm, csv1, name_of_csv):
    list0 = []
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []
    list8 = []
    list9 = []
    list10 = []
    list11 = []
    list12 = []
    list13 = []
    list14 = []
    list15 = []
    corr = 0
    for line, r, n, s in zip(csv1, rf, nb, svm):
        list0.append(line[0])
        list1.append(line[1])
        list2.append(line[2])
        list3.append(line[3])
        list4.append(line[4])
        list5.append(line[5])
        list6.append(line[6])
        list7.append(line[7])
        list8.append(line[8])
        list9.append(line[9])
        list10.append(line[10])
        list11.append(line[11])

        list15.append(line[12])

        if r == 0:
            i = 0
        elif r == 1:
            i = 1
        else:
            if line[10] == '1':
                i = 1
            else:
                i = 0
        list12.append(i)

        if n == 0:
            i = 0
        elif n == 1:
            i = 1
        else:
            if line[10] == '1':
                i = 1
            else:
                i = 0
        list13.append(i)

        if s == 0:
            i = 0
        elif s == 1:
            i = 1
        else:
            if line[10] == '1':
                i = 1
            else:
                i = 0
        list14.append(i)
    name_dict = {
        'Hotel_name': list0,
        'Hotel_chain': list1,
        'Guest_country': list2,
        'Room_info': list3,
        'Nights_stayed': list4,
        'Date of stay': list5,
        'Travel_type': list6,
        'Review': list7,
        'Grade': list8,
        'Title': list9,
        'Positive': list10,
        'CRNN': list11,
        'Random Forest': list12,
        'Naive Bayes': list13,
        'SVM': list14,
        'Facilities': list15,
    }
    df = pd.DataFrame(name_dict)
    df.to_csv(name_of_csv, index=False)


add_column_to_csv_cat(ahpc_rf, ahpc_nb, ahpc_svm, open("../all_hotels_predict_categorized.csv", "r"), 'all_hotels_categorized(nn_ml).svc')
add_column_to_csv_row(ahpr_rf, ahpr_nb, ahpr_svm, open("../all_hotels_predict_row.csv", "r"), 'all_hotels_row(nn_ml).svc')
