import pickle
import re
import pandas as pd
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def remove_stopwords(text):
    x = []

    for i in text.split():
        if i not in stopwords.words('english'):
            x.append(i)

    return x


def stem_words(text):
    x = ''

    for i in text:
        x += ps.stem(i)
        x += ' '
    return x


clf = pickle.load(open("Pickle Files/clf.pkl", "rb"))
cv = pickle.load(open("Pickle Files/cv.pkl", "rb"))


def sentiment_analyzer(txt):
    text = pd.Series([txt])
    text = text.str.replace('[^A-Za-z0-9]', ' ', flags=re.UNICODE, regex=True)
    text = text.str.lower()
    text = text.apply(remove_stopwords)
    text = text.apply(stem_words)

    vector = cv.transform(text).toarray()
    prediction = clf.predict(vector)

    if prediction == 1:
        return "Positive"
    elif prediction == 0:
        return "Neutral"
    elif prediction == -1:
        return 'Negative'
    else:
        return "Irrelevant"
