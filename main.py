import nltk
import spacy
import numpy as np
import pandas
import string
import gensim
import re
# import IPython
from spacy import displacy
from nltk import pos_tag
from nltk.corpus import stopwords
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


# Adds noun to dictionary with frequency and document source index
def addNounToDict(noun, dictionary, indexSource):
    if noun in dictionary:
        dictionary[noun][0] = dictionary[noun][0] + 1
        dictionary[noun][1].add(indexSource)
    else:
        dictionary[noun] = [1, set()]
        dictionary[noun][1].add(indexSource)


# Parses dependencies for noun in document and returns descriptors
def getDescriptors(noun, documentNumber):
    descriptors = set()
    instance = initialDF.body[documentNumber]
    parsedTokens = depParser(instance)
    for token in parsedTokens:
        if stemmer.stem(token.head.text) == noun:
            if token.dep_ == "amod":
                descriptors.add(token.text)
    return descriptors


# Load in dependency parser
print("loading dependency parser...")
print("Make sure 'en_core_web_sm' is downloaded from spacy, can use '/python -m spacy download en_core_web_sm'")
depParser = spacy.load("en_core_web_sm")

# Read in data
print("reading in data...")
initialDF = pandas.read_csv("reddit_pfizer_vaccine.csv", usecols=['body'])
initialDF.dropna(inplace=True)
initialDF.reset_index(inplace=True, drop=True)

# Go through DF and retrieve all instances of nouns from text, with DF indices appended
print("retrieving noun instances...")
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
print("sorting noun dictionary...")
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

# Get all modifiers for every instance of word
print("retrieving modifiers...")
selectedNoun = descNouns[inputWord]
print(selectedNoun)
nounInstances = selectedNoun[1]
modifiers = set()
for instance in nounInstances:
    modifiers.update(getDescriptors(inputWord, instance))
modifiers = list(modifiers)
print(modifiers)
