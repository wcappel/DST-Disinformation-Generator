import nltk
import spacy
import numpy as np
import tensorflow as tf
import pandas
import string
# import IPython
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

# tokens = nltk.word_tokenize(sentence1)
# print(tokens)
# tagged = nltk.pos_tag(tokens)
# print(tagged)

stemmer = nltk.PorterStemmer()
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

# Sort noun dictionary by frequency
descNouns = {key: val for key, val in sorted(nouns.items(), key=lambda element: element[1], reverse=True)}
# print(descNouns)

# Get top ~25 words referenced and filter out punct.
mostRef = list(descNouns.keys())[0:24]
for word in mostRef:
    if len(word) <= 1:
        mostRef.remove(word)

# Get user input for word to be used
print("Most referenced nouns/entities: " + str(mostRef))
inputWord = ""
while True:
    inputWord = input("Enter the word from the list you wish to use: ")
    if type(inputWord) is str:
        inputWord = inputWord.lower()
        if inputWord in mostRef:
            break
        else:
            print("Please pick a word from the list.")
    else:
        print("Please pick a word from the list.")

selectedNoun = descNouns[inputWord]
# print(selectedNoun)
nounInstances = selectedNoun[1]
# for instance in nounInstances:
