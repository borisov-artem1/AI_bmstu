import re

import nltk
import pymorphy2
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')

morph = pymorphy2.MorphAnalyzer()


def getNormalFormWord(word):
    return morph.parse(word)[0].normal_form


def delUselessSigns(phrase):
    return re.sub("[^а-яa-z0-9'№ -]", "", phrase)


def getNormalFormPhrase(phrase):
    wordArr = word_tokenize(phrase, language="english")
    return ' '.join(getNormalFormWord(word) for word in wordArr)


def toLower(phrase):
    return phrase.lower()


def preprocessing(phrase):
    phrase = toLower(phrase)
    phrase = delUselessSigns(phrase)
    return getNormalFormPhrase(phrase)


def updateType(input):
    dict = {"Protein": 0, "Gainer": 0.2,
            "Mass protein": 0.4, "Casein": 0.6,
            "Whey": 0.8, "Isolate": 1}
    return dict[input]


def updateIntensity(input):
    intensity_dict = {
    "extremely low": 0, "low": 0.25,
    "medium": 0.5, "high": 0.75,
    "extremely high": 1 }
    return intensity_dict[input]
