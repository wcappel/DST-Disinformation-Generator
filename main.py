import nltk
import spacy
import numpy as np
import pandas
import string
import gensim
import re
import ssl
# import IPython
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import CFG
from nltk.parse.generate import generate
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('vader_lexicon')


# Adds noun to dictionary with frequency and document source index
def addNounToDict(noun, dictionary, indexSource):
    if noun in dictionary:
        dictionary[noun][0] = dictionary[noun][0] + 1
        dictionary[noun][1].add(indexSource)
    else:
        dictionary[noun] = [1, set()]
        dictionary[noun][1].add(indexSource)


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
# Also formats sentences for word2vec
print("retrieving noun instances...")
stemmer = nltk.PorterStemmer()
stopWords = stopwords.words()
nouns = {}
allSentences = []
modSentences = []

for index, row in initialDF.iterrows():
    row = row.astype(str)
    text = row['body']
    rowTokens = nltk.word_tokenize(text)
    docSentences = nltk.sent_tokenize(text)
    for sentence in docSentences:
        processedSentence = []
        sentence = "".join([char for char in sentence if char not in string.punctuation])
        for word in sentence.split(" "):
            if word not in stopWords:
                processedSentence.append(stemmer.stem(word))
        allSentences.append(processedSentence)
        sentenceDep = depParser(sentence)
        hasModifier = False
        for token in sentenceDep:
            if token.dep_ == "amod":
                hasModifier = True
        if hasModifier:
            modSentences.append(sentence)
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
mostRef = list(descNouns.keys())[0:25]
refDocs = list(descNouns.values())[0:25]
refDocs = [x[1] for x in refDocs]

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

# Use nltk sentiment analysis to get instances of neg/pos sentences w/ modifiers
sa = SentimentIntensityAnalyzer()
negSentiment = []
posSentiment = []
for sentence in modSentences:
    sentencePolarity = sa.polarity_scores(sentence)
    if sentencePolarity['compound'] >= 0.7:
        posSentiment.append(sentence)
    elif sentencePolarity['compound'] <= -0.7:
        negSentiment.append(sentence)
print(negSentiment)

# Pos tag and convert Noun Phrases to CFG

# Use two embedding matrices to harness descriptors for pos. and neg.?

# Create CFG rules w/ specific noun and desired descriptors

# Generate w/ CFG