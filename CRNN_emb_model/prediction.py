from tensorflow import keras
import csv
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MAX_LEN = 24
NB_WORDS = 10000
model = keras.models.load_model('emb_model')
tk = Tokenizer(num_words=NB_WORDS,
               filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n',
               lower=True,
               split=" ")

country_hall = open("../hotels-parsed-added-facilities/country_hall(parsed)(1)_facilities.csv", "r")
leeds = open("../hotels-parsed-added-facilities/leeds(parsed)(1)_facilities.csv", "r")
park_royal = open("../hotels-parsed-added-facilities/park_royal(parsed1)_facilities.csv", "r")
riverbank = open("../hotels-parsed-added-facilities/riverbank(parsed1)_facilities.csv", "r")
victoria_london = open("../hotels-parsed-added-facilities/victoria_london(parsed)_facilities.csv", "r")
victoria = open("../hotels-parsed-added-facilities/victoria(parsed)(1)_facilities.csv", "r")
vondelpark = open("../hotels-parsed-added-facilities/vondelpark(parsed)(1)_facilities.csv", "r")
westminster = open("../hotels-parsed-added-facilities/westminster(parsed1)_facilities.csv", "r")
wroclaw = open("../hotels-parsed-added-facilities/wrocław(parsed1)_facilities.csv", "r")
dubai = open("../hotels-parsed-added-facilities/Radisson_Blu_Dubai_facilities.csv", "r")
edinburgh = open("../hotels-parsed-added-facilities/Radisson_Blu_ Edinburgh_facilities.csv", "r")
edwardian = open("../hotels-parsed-added-facilities/Radisson_Blu_Edwardian_facilities.csv", "r")
edwardian_london = open("../hotels-parsed-added-facilities/Radisson_Blu_Edwardian_London_facilities.csv", "r")
glasgow = open("../hotels-parsed-added-facilities/Radisson_Blu_Glasgow_facilities.csv", "r")
liverpool = open("../hotels-parsed-added-facilities/Radisson_Blu_Liverpool_facilities.csv", "r")
manchester = open("../hotels-parsed-added-facilities/Radisson_Blu_Manchester_facilities.csv", "r")
sydney = open("../hotels-parsed-added-facilities/Radisson_Blu_Plaza Hotel Sydney_facilities.csv", "r")

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
    return review, sentiment


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
    prepare1 = pd.Series()
    i = 0
    for review in reviews:
        series = pd.Series(review, index=[i])
        prepare1 = prepare1.append(series)
        i = i + 1

    tk.fit_on_texts(prepare1)
    prepare2 = tk.texts_to_sequences(prepare1)
    prepared = pad_sequences(prepare2, maxlen=MAX_LEN)

    prediction = model.predict(prepared)
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
    list11 = []
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
        list10.append(line[10])
        if pred == 0:
            i = 0
            # list11.append(0)
        else:
            i = 1
            # list11.append(1)
        list11.append(i)
        if str(i) == line[9]:
            corr = corr + 1
    print(corr)
    print(len(predicted))
    print(corr / len(predicted))
    name_dict = {
        'Guest_country': list0,
        'Room_info': list1,
        'Nights_stayed': list2,
        'Date of stay': list3,
        'Travel_type': list4,
        'Review': list5,
        'Grade': list6,
        'Title': list7,
        'Negative': list8,
        'Positive': list9,
        'Facilities': list10,
        'Predicted': list11
    }
    df = pd.DataFrame(name_dict)
    df.to_csv(name_of_csv)


add_column_to_csv(predicted_country_hall,
                  csv.reader(open("../hotels-parsed-added-facilities/country_hall(parsed)(1)_facilities.csv", "r")),
                  'predicted_country_hall.csv')
add_column_to_csv(predicted_leeds,
                  csv.reader(open("../hotels-parsed-added-facilities/leeds(parsed)(1)_facilities.csv", "r")),
                  'predicted_leeds.csv')
add_column_to_csv(predicted_park_royal,
                  csv.reader(open("../hotels-parsed-added-facilities/park_royal(parsed1)_facilities.csv", "r")),
                  'predicted_park_royal.csv')
add_column_to_csv(predicted_riverbank,
                  csv.reader(open("../hotels-parsed-added-facilities/riverbank(parsed1)_facilities.csv", "r")),
                  'predicted_riverbank.csv')
add_column_to_csv(predicted_victoria_london,
                  csv.reader(open("../hotels-parsed-added-facilities/victoria_london(parsed)_facilities.csv", "r")),
                  'predicted_victoria_london.csv')
add_column_to_csv(predicted_victoria,
                  csv.reader(open("../hotels-parsed-added-facilities/victoria(parsed)(1)_facilities.csv", "r")),
                  'predicted_victoria.csv')
add_column_to_csv(predicted_vondelpark,
                  csv.reader(open("../hotels-parsed-added-facilities/vondelpark(parsed)(1)_facilities.csv", "r")),
                  'predicted_vondelpark.csv')
add_column_to_csv(predicted_westminster,
                  csv.reader(open("../hotels-parsed-added-facilities/westminster(parsed1)_facilities.csv", "r")),
                  'predicted_westminster.csv')
add_column_to_csv(predicted_wroclaw,
                  csv.reader(open("../hotels-parsed-added-facilities/wrocław(parsed1)_facilities.csv", "r")),
                  'predicted_wroclaw.csv')
add_column_to_csv(predicted_dubai,
                  csv.reader(open("../hotels-parsed-added-facilities/Radisson_Blu_Dubai_facilities.csv", "r")),
                  'predicted_dubai.csv')
add_column_to_csv(predicted_edinburg,
                  csv.reader(open("../hotels-parsed-added-facilities/Radisson_Blu_ Edinburgh_facilities.csv", "r")),
                  'predicted_edinburg.csv')
add_column_to_csv(predicted_edwardian,
                  csv.reader(open("../hotels-parsed-added-facilities/Radisson_Blu_Edwardian_facilities.csv", "r")),
                  'predicted_edwardian.csv')
add_column_to_csv(predicted_edwardian_london, csv.reader(
    open("../hotels-parsed-added-facilities/Radisson_Blu_Edwardian_London_facilities.csv", "r")),
                  'predicted_edwardian_london.csv')
add_column_to_csv(predicted_glasgow,
                  csv.reader(open("../hotels-parsed-added-facilities/Radisson_Blu_Glasgow_facilities.csv", "r")),
                  'predicted_glasgow.csv')
add_column_to_csv(predicted_liverpool,
                  csv.reader(open("../hotels-parsed-added-facilities/Radisson_Blu_Liverpool_facilities.csv", "r")),
                  'predicted_liverpool.csv')
add_column_to_csv(predicted_manchester,
                  csv.reader(open("../hotels-parsed-added-facilities/Radisson_Blu_Manchester_facilities.csv", "r")),
                  'predicted_manchester.csv')
add_column_to_csv(predicted_sydney, csv.reader(
    open("../hotels-parsed-added-facilities/Radisson_Blu_Plaza Hotel Sydney_facilities.csv", "r")),
                  'predicted_sydney.csv')
