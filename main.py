import nltk
import spacy
import numpy as np
import pandas
import string
import random
import sys
from nltk import CFG
from nltk.corpus import stopwords
from nltk.parse.generate import generate
from nltk.sentiment import SentimentIntensityAnalyzer
from spacy.matcher import Matcher
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
punct = string.punctuation

for index, row in initialDF.iterrows():
    row = row.astype(str)
    text = row['body']
    rowTokens = nltk.word_tokenize(text)
    docSentences = nltk.sent_tokenize(text)
    for sentence in docSentences:
        sentence = "".join([char for char in sentence if char not in punct])
        allSentences.append(sentence.lower())
    rowTags = nltk.pos_tag(rowTokens)
    for tag in rowTags:
        if tag[1] == "NN" or tag[1] == "NNS" or tag[1] == "NNP" or tag[1] == "NNPS":
            if tag[0] not in stopWords and len(tag[0]) <= 12:
                removedPunct = "".join([char for char in tag[0] if char not in punct])
                stemmedNoun = stemmer.stem(removedPunct)
                addNounToDict(stemmedNoun, nouns, index)
    # Put and adjust pos tagging step in per-sentence loop


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
negSentences = set()
posSentences = set()
negModifiers = set()
posModifiers = set()
for sentence in allSentences:
    sentencePolarity = sa.polarity_scores(sentence)
    if sentencePolarity['compound'] >= 0.8:
        posSentences.add(sentence)
        sentenceDep = depParser(sentence)
        for token in sentenceDep:
            if token.dep_ == "amod":
                posModifiers.add(token.text.lower())
    elif sentencePolarity['compound'] <= -0.8:
        negSentences.add(sentence)
        sentenceDep = depParser(sentence)
        for token in sentenceDep:
            if token.dep_ == "amod":
                negModifiers.add(token.text.lower())

# Patterns for verb phrases
VP = [[{"POS": "VERB"}, {"POS": "PRON"}, {"POS": "VERB"}, {"POS": "ADJ"}],
      [{"POS": "VERB"}, {"POS": "PRON"}, {"POS": "VERB"}], [{"POS": "VERB"}, {"POS": "NOUN"}, {"POS": "NOUN"}],
      [{"POS": "VERB"}, {"POS": "DET"}, {"POS": "NOUN"}], [{"POS": "VERB"}, {"POS": "DET"}, {"POS": "NOUN"}, {"POS": "NOUN"}],
      [{"POS": "VERB"}, {"POS": "PRON"}], [{"POS": "VERB"}, {"POS": "PRON"}], [{"POS": "VERB"}, {"POS": "PRON"}, {"POS": "VERB"}],
      [{"POS": "VERB"}, {"POS": "PRON"}, {"POS": "VERB"}, {"POS": "ADJ"}]]

VR = [[{"POS": "VERB"}, {"POS": "ADP"}, {"POS": "NOUN"}], [{"POS": "VERB"}, {"POS": "ADP"}, {"POS": "PRON"}],
       [{"POS": "VERB"}, {"POS": "ADP"}, {"POS": "DET"}, {"POS": "NOUN"}],
       [{"POS": "VERB"}, {"POS": "ADP"}, {"POS": "ADJ"}, {"POS": "NOUN"}],
       [{"POS": "VERB"}, {"POS": "ADP"}, {"POS": "NOUN"}, {"POS": "NOUN"}]]

vpMatcher = Matcher(depParser.vocab)
vpMatcher.add("VP", VP)
vrMatcher = Matcher(depParser.vocab)
vrMatcher.add("VR", VR)

# Extract verb phrases from negative and positive sentences
negVP = set()
negVR = set()
for sentence in list(negSentences):
    sentence = depParser(sentence)
    vpMatches = vpMatcher(sentence)
    vrMatches = vrMatcher(sentence)
    for match_id, start, end in vpMatches:
        span = sentence[start:end]
        negVP.add(span.text)
    for match_id, start, end in vrMatches:
        span = sentence[start:end]
        negVR.add(span.text)

print(list(negVP))
print(list(negVR))

# Create CFG rules w/ specific noun and desired descriptors
grammarString = """
S -> NP VP
NP -> Det Nom
Nom -> N | Adj N
VP -> V Adj | V | V 
V -> 'is' | 'was'
Det -> 'the' | 'that'
"""
grammarString += "\nN -> '" + inputWord + "'"
posString = grammarString
negString = grammarString
for mod in posModifiers:
    posString += "\nAdj -> '" + mod + "'"

for mod in negModifiers:
    negString += "\nAdj -> '" + mod + "'"

posGrammar = CFG.fromstring(posString)
negGrammar = CFG.fromstring(negString)

# Generate sentences w/ CFG rules
generatedNeg = list(generate(negGrammar, n=sys.maxsize))
random.shuffle(generatedNeg)
scoredNeg = set()
for sentence in generatedNeg:
    sentence = ' '.join(sentence)
    if sa.polarity_scores(sentence)['compound'] <= -0.6:
        scoredNeg.add(sentence)
print(list(scoredNeg))

# Pos tag and convert Noun Phrases to CFG?

# Use two embedding matrices to harness descriptors for pos. and neg.? Yes, probably do this.
