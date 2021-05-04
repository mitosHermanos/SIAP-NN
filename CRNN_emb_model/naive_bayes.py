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

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = naive_bayes.MultinomialNB()
clf.fit(X_train_tfidf, y_train)



country_hall = open("../hotels-with-categorized-facilities/country_hall(parsed)("
                    "1)_facilities_categorized_facilities.csv", "r")
leeds = open("../hotels-with-categorized-facilities/leeds(parsed)(1)_facilities_categorized_facilities.csv", "r")
park_royal = open("../hotels-with-categorized-facilities/park_royal(parsed1)_facilities_categorized_facilities.csv", "r")
riverbank = open("../hotels-with-categorized-facilities/riverbank(parsed1)_facilities_categorized_facilities.csv", "r")
victoria_london = open("../hotels-with-categorized-facilities/victoria_london("
                       "parsed)_facilities_categorized_facilities.csv", "r")
victoria = open("../hotels-with-categorized-facilities/victoria(parsed)(1)_facilities_categorized_facilities.csv", "r")
vondelpark = open("../hotels-with-categorized-facilities/vondelpark(parsed)(1)_facilities_categorized_facilities.csv", "r")
westminster = open("../hotels-with-categorized-facilities/westminster(parsed1)_facilities_categorized_facilities.csv", "r")
wroclaw = open("../hotels-with-categorized-facilities/wrocław(parsed1)_facilities_categorized_facilities.csv", "r")
dubai = open("../hotels-with-categorized-facilities/Radisson_Blu_Dubai_facilities_categorized_facilities.csv", "r")
edinburgh = open("../hotels-with-categorized-facilities/Radisson_Blu_ Edinburgh_facilities_categorized_facilities.csv", "r")
edwardian = open("../hotels-with-categorized-facilities/Radisson_Blu_Edwardian_facilities_categorized_facilities.csv", "r")
edwardian_london = open("../hotels-with-categorized-facilities"
                        "/Radisson_Blu_Edwardian_London_facilities_categorized_facilities.csv", "r")
glasgow = open("../hotels-with-categorized-facilities/Radisson_Blu_Glasgow_facilities_categorized_facilities.csv", "r")
liverpool = open("../hotels-with-categorized-facilities/Radisson_Blu_Liverpool_facilities_categorized_facilities.csv", "r")
manchester = open("../hotels-with-categorized-facilities/Radisson_Blu_Manchester_facilities_categorized_facilities.csv", "r")
sydney = open("../hotels-with-categorized-facilities/Radisson_Blu_Plaza Hotel "
              "Sydney_facilities_categorized_facilities.csv", "r")

country_hall_csv = csv.reader(country_hall)
leeds_csv = csv.reader(leeds)
park_royal_csv = csv.reader(park_royal)
riverbank_csv = csv.reader(riverbank)
victoria_london_csv = csv.reader(victoria_london)
victoria_csv = csv.reader(victoria)
vondelpark_csv = csv.reader(vondelpark)
westminster_csv = csv.reader(westminster)
wroclaw_csv = csv.reader(wroclaw)
dubai_csv = csv.reader(dubai)
edinburgh_csv = csv.reader(edinburgh)
edwardian_csv = csv.reader(edwardian)
edwardian_london_csv = csv.reader(edwardian_london)
glasgow_csv = csv.reader(glasgow)
liverpool_csv = csv.reader(liverpool)
manchester_csv = csv.reader(manchester)
sydney_csv = csv.reader(sydney)


def get_values_from_csv(csv):
    review = []
    sentiment = []
    for line in csv:
        review.append(line[5])
        if line[9] == 1:
            sentiment.append(1)
        else:
            sentiment.append(0)
    return review[1:], sentiment[1:]


review_country_hall = get_values_from_csv(country_hall_csv)[0]
sentiment_country_hall = get_values_from_csv(country_hall_csv)[1]

review_leeds = get_values_from_csv(leeds_csv)[0]
sentiment_leeds = get_values_from_csv(leeds_csv)[1]

review_park_royal = get_values_from_csv(park_royal_csv)[0]
sentiment_park_royal = get_values_from_csv(park_royal_csv)[1]

review_riverbank = get_values_from_csv(riverbank_csv)[0]
sentiment_riverbank = get_values_from_csv(riverbank_csv)[1]

review_victoria_london = get_values_from_csv(victoria_london_csv)[0]
sentiment_victoria_london = get_values_from_csv(victoria_london_csv)[1]

review_victoria = get_values_from_csv(victoria_csv)[0]
sentiment_victoria = get_values_from_csv(victoria_csv)[1]

review_vondelpark = get_values_from_csv(vondelpark_csv)[0]
sentiment_vondelpark = get_values_from_csv(vondelpark_csv)[1]

review_westminster = get_values_from_csv(westminster_csv)[0]
sentiment_westminster = get_values_from_csv(westminster_csv)[1]

review_wroclaw = get_values_from_csv(wroclaw_csv)[0]
sentiment_wroclaw = get_values_from_csv(wroclaw_csv)[1]

review_dubai = get_values_from_csv(dubai_csv)[0]
sentiment_dubai = get_values_from_csv(dubai_csv)[1]

review_edinburgh = get_values_from_csv(edinburgh_csv)[0]
sentiment_edinburgh = get_values_from_csv(edinburgh_csv)[1]

review_edwardian = get_values_from_csv(edwardian_csv)[0]
sentiment_edwardian = get_values_from_csv(edwardian_csv)[1]

review_edwardian_london = get_values_from_csv(edwardian_london_csv)[0]
sentiment_edwardian_london = get_values_from_csv(edwardian_london_csv)[1]

review_glasgow = get_values_from_csv(glasgow_csv)[0]
sentiment_glasgow = get_values_from_csv(glasgow_csv)[1]

review_liverpool = get_values_from_csv(liverpool_csv)[0]
sentiment_liverpool = get_values_from_csv(liverpool_csv)[1]

review_manchester = get_values_from_csv(manchester_csv)[0]
sentiment_manchester = get_values_from_csv(manchester_csv)[1]

review_sydney = get_values_from_csv(sydney_csv)[0]
sentiment_sydney = get_values_from_csv(country_hall_csv)[1]


def get_predictions(reviews):
    prepare1 = pd.Series(dtype=object)
    i = 0
    for review in reviews:
        series = pd.Series(review, index=[i])
        prepare1 = prepare1.append(series)
        i = i + 1

    X_train_counts = count_vect.transform(prepare1)

    prediction = clf.predict(X_train_counts)
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


predicted_country_hall = get_predictions(review_country_hall)
predicted_leeds = get_predictions(review_leeds)
predicted_park_royal = get_predictions(review_park_royal)
predicted_riverbank = get_predictions(review_riverbank)
predicted_victoria_london = get_predictions(review_victoria_london)
predicted_victoria = get_predictions(review_victoria)
predicted_vondelpark = get_predictions(review_vondelpark)
predicted_westminster = get_predictions(review_westminster)
predicted_wroclaw = get_predictions(review_wroclaw)
predicted_dubai = get_predictions(review_dubai)
predicted_edinburg = get_predictions(review_edinburgh)
predicted_edwardian = get_predictions(review_edwardian)
predicted_edwardian_london = get_predictions(review_edwardian_london)
predicted_glasgow = get_predictions(review_glasgow)
predicted_liverpool = get_predictions(review_liverpool)
predicted_manchester = get_predictions(review_manchester)
predicted_sydney = get_predictions(review_sydney)


def add_column_to_csv(predicted, csv1, name_of_csv):
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
    corr = 0
    for line, pred in zip(csv1, predicted):
        # print(pred)
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
        if pred == 0:
            i = 0
        elif pred == 1:
            i = 1
        else:
            if line[8] == '1':
                i = 1
            else:
                i = 0
        list10.append(i)
        if str(i) == line[8]:
            corr = corr + 1
    print(corr)
    print(len(predicted))
    print(corr / len(predicted))
    name_dict = {
        'Guest_country': list0[1:],
        'Room_info': list1[1:],
        'Nights_stayed': list2[1:],
        'Date of stay': list3[1:],
        'Travel_type': list4[1:],
        'Review': list5[1:],
        'Grade': list6[1:],
        'Title': list7[1:],
        'Positive': list8[1:],
        'Facilities': list9[1:],
        'Predicted': list10[1:]
    }
    df = pd.DataFrame(name_dict)
    # df.to_csv(name_of_csv, index=False)


add_column_to_csv(predicted_country_hall,
                  csv.reader(open("../hotels-with-categorized-facilities/country_hall(parsed)("
                                  "1)_facilities_categorized_facilities.csv", "r")),
                  'predicted_nb_country_hall.csv')
add_column_to_csv(predicted_leeds,
                  csv.reader(open("../hotels-with-categorized-facilities/leeds(parsed)("
                                  "1)_facilities_categorized_facilities.csv", "r")),
                  'predicted_nb_leeds.csv')
add_column_to_csv(predicted_park_royal,
                  csv.reader(open("../hotels-with-categorized-facilities/park_royal("
                                  "parsed1)_facilities_categorized_facilities.csv", "r")),
                  'predicted_nb_park_royal.csv')
add_column_to_csv(predicted_riverbank,
                  csv.reader(open("../hotels-with-categorized-facilities/riverbank("
                                  "parsed1)_facilities_categorized_facilities.csv", "r")),
                  'predicted_nb_riverbank.csv')
add_column_to_csv(predicted_victoria_london,
                  csv.reader(open("../hotels-with-categorized-facilities/victoria_london("
                                  "parsed)_facilities_categorized_facilities.csv", "r")),
                  'predicted_nb_victoria_london.csv')
add_column_to_csv(predicted_victoria,
                  csv.reader(open("../hotels-with-categorized-facilities/victoria(parsed)("
                                  "1)_facilities_categorized_facilities.csv", "r")),
                  'predicted_nb_victoria.csv')
add_column_to_csv(predicted_vondelpark,
                  csv.reader(open("../hotels-with-categorized-facilities/vondelpark(parsed)("
                                  "1)_facilities_categorized_facilities.csv", "r")),
                  'predicted_nb_vondelpark.csv')
add_column_to_csv(predicted_westminster,
                  csv.reader(open("../hotels-with-categorized-facilities/westminster("
                                  "parsed1)_facilities_categorized_facilities.csv", "r")),
                  'predicted_nb_westminster.csv')
add_column_to_csv(predicted_wroclaw,
                  csv.reader(open("../hotels-with-categorized-facilities/wrocław("
                                  "parsed1)_facilities_categorized_facilities.csv", "r")),
                  'predicted_nb_wroclaw.csv')
add_column_to_csv(predicted_dubai,
                  csv.reader(open("../hotels-with-categorized-facilities"
                                  "/Radisson_Blu_Dubai_facilities_categorized_facilities.csv", "r")),
                  'predicted_nb_dubai.csv')
add_column_to_csv(predicted_edinburg,
                  csv.reader(open("../hotels-with-categorized-facilities/Radisson_Blu_ "
                                  "Edinburgh_facilities_categorized_facilities.csv", "r")),
                  'predicted_nb_edinburg.csv')
add_column_to_csv(predicted_edwardian,
                  csv.reader(open("../hotels-with-categorized-facilities"
                                  "/Radisson_Blu_Edwardian_facilities_categorized_facilities.csv", "r")),
                  'predicted_nb_edwardian.csv')
add_column_to_csv(predicted_edwardian_london, csv.reader(
    open("../hotels-with-categorized-facilities/Radisson_Blu_Edwardian_London_facilities_categorized_facilities.csv", "r")),
                  'predicted_nb_edwardian_london.csv')
add_column_to_csv(predicted_glasgow,
                  csv.reader(open("../hotels-with-categorized-facilities"
                                  "/Radisson_Blu_Glasgow_facilities_categorized_facilities.csv", "r")),
                  'predicted_nb_glasgow.csv')
add_column_to_csv(predicted_liverpool,
                  csv.reader(open("../hotels-with-categorized-facilities"
                                  "/Radisson_Blu_Liverpool_facilities_categorized_facilities.csv", "r")),
                  'predicted_nb_liverpool.csv')
add_column_to_csv(predicted_manchester,
                  csv.reader(open("../hotels-with-categorized-facilities"
                                  "/Radisson_Blu_Manchester_facilities_categorized_facilities.csv", "r")),
                  'predicted_nb_manchester.csv')
add_column_to_csv(predicted_sydney, csv.reader(
    open("../hotels-with-categorized-facilities/Radisson_Blu_Plaza Hotel Sydney_facilities_categorized_facilities.csv", "r")),
                  'predicted_nb_sydney.csv')
