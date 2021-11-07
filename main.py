import nltk
import spacy
import numpy as np
import tensorflow as tf
import pandas
import string
from spacy import displacy
from tensorflow import keras
from nltk import pos_tag
from nltk.corpus import stopwords

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


def addNounToDict(noun, dictionary, indexSource):
    if noun in dictionary:
        dictionary[noun][0] = dictionary[noun][0] + 1
        dictionary[noun][1].add(indexSource)
    else:
        dictionary[noun] = [1, set()]
        dictionary[noun][1].add(indexSource)


# Load in dependency parser
depParser = spacy.load("en_core_web_sm")

# Examples to think about
sentence1 = "That shitty car."
sentence2 = "This car is shitty."

parsed1 = depParser(sentence1)
parsed2 = depParser(sentence2)

# Display parsed dependencies
# splacy.serve(parsed1)
# displacy.serve(parsed2)

# Read in data
initialDF = pandas.read_csv("reddit_pfizer_vaccine.csv", usecols=['body'])
# print(initialDF)

# tokens = nltk.word_tokenize(sentence1)
# print(tokens)
# tagged = nltk.pos_tag(tokens)
# print(tagged)
stemmer = nltk.PorterStemmer()
# np = stemmer.stem("bitcoins")
# ns = stemmer.stem("bitcoin")


stopWords = stopwords.words()
nouns = {}
for index, row in initialDF.iterrows():
    row = row.astype(str)
    text = row['body']
    rowTokens = nltk.word_tokenize(text)
    rowTags = nltk.pos_tag(rowTokens)
    for tag in rowTags:
        if tag[1] == "NN" or tag[1] == "NNS" or tag[1] == "NNP" or tag[1] == "NNPS":
            if tag[0] not in stopWords and len(tag[0]) <= 12:
                removedPunct = "".join([char for char in tag[0] if char not in string.punctuation])
                stemmedNoun = stemmer.stem(removedPunct)
                addNounToDict(stemmedNoun, nouns, index)

descNouns = {key: val for key, val in sorted(nouns.items(), key=lambda element: element[1], reverse=True)}
print(descNouns)