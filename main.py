import nltk
import spacy
import pandas
import string
import random
from nltk import CFG, PCFG
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
spacyModel = spacy.load("en_core_web_sm")

# Read in data
print("reading in data...")
initialDF = pandas.read_csv("reddit_pfizer_vaccine.csv", usecols=['body'])
initialDF.dropna(inplace=True)
initialDF.reset_index(inplace=True, drop=True)

# Go through 'body' column of DF and retrieve all instances of nouns from text, with DF indices appended
# Also formats sentences for sentiment analysis and further PoS tagging for phrase extraction
print("retrieving noun and named entity instances...")
stemmer = nltk.PorterStemmer()
stopWords = stopwords.words()
nouns = {}
namedEntities = {}
allSentences = set()
modSentences = []
punct = string.punctuation

bodyColumn = initialDF['body'].astype(str).to_list()
for index, row in enumerate(bodyColumn):
    text = row
    docSentences = nltk.sent_tokenize(text)
    for sentence in docSentences:
        sentenceDoc = spacyModel(sentence)
        sentence = "".join([char for char in sentence if char not in punct])
        allSentences.add(sentence.lower())
        for token in sentenceDoc:
            if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                if len(token.text[0]) <= 12:
                    addNounToDict(token.text.lower(), nouns, index)
        for ne in sentenceDoc.ents:
            addNounToDict(ne.text.lower(), namedEntities, index)

# Sort noun dictionary by frequency
print("sorting noun dictionaries...")
descNouns = {key: val for key, val in sorted(nouns.items(), key=lambda element: element[1], reverse=True)}
descNE = {key: val for key, val in sorted(namedEntities.items(), key=lambda element: element[1], reverse=True)}

# Get top ~25 words and entities referenced and filter out punct.
mostRefNouns = list(descNouns.keys())[0:25]
refNounsDocs = list(descNouns.values())[0:25]
refNounsDocs = [x[1] for x in refNounsDocs]

mostRefNE = list(descNE.keys())[0:25]
refNEDocs = list(descNE.values())[0:25]
refNEDocs = [x[1] for x in refNEDocs]

for word in mostRefNouns:
    if len(word) <= 1:
        mostRefNouns.remove(word)

for word in mostRefNE:
    if len(word) <= 1:
        mostRefNE.remove(word)

# Get user input for word to be used
print("Most referenced nouns: \n\t" + str(mostRefNouns) + "\n Most referenced entities: \n\t" + str(mostRefNE))
inputWord = ""
while True:
    inputWord = input("Enter the word from either list that you wish to use: ")
    if type(inputWord) is str:
        inputWord = inputWord.lower()
        if inputWord in mostRefNouns or inputWord in mostRefNE:
            break
        else:
            print("Please pick a word from either list.")
    else:
        print("Please pick a word from either list.")

# Use nltk sentiment analysis to get instances of neg/pos sentences w/ modifiers
print("running SA and retrieving modifiers...")
sa = SentimentIntensityAnalyzer()
allSentences = list(allSentences)
negSentences = set()
posSentences = set()
negModifiers = set()
posModifiers = set()
for sentence in allSentences:
    sentencePolarity = sa.polarity_scores(sentence)
    if sentencePolarity['compound'] >= 0.4:
        posSentences.add(sentence)
        sentenceDep = spacyModel(sentence)
        for token in sentenceDep:
            if token.dep_ == "amod":
                posModifiers.add(token.text.lower())
    elif sentencePolarity['compound'] <= -0.5:
        negSentences.add(sentence)
        sentenceDep = spacyModel(sentence)
        for token in sentenceDep:
            if token.dep_ == "amod":
                negModifiers.add(token.text.lower())

# Patterns for verb phrases and preposition phrases
VP = [[{"POS": "ADV", "OP": "?"}, {"POS": "VERB", "OP": "+"}, {"POS": "PRON"}, {"POS": "VERB", "OP": "*"}, {"POS": "ADJ", "OP": "*"}, {"POS": "NOUN", "OP": "?"}, {"POS": "PRON", "OP": "?"}],
      [{"POS": "ADV", "OP": "?"}, {"POS": "VERB", "OP": "+"}, {"POS": "DET", "OP": "*"}, {"POS": "ADJ", "OP": "*"}, {"POS": "NOUN", "OP": "+"}]]

PP = [[{"POS": "ADP"}, {"POS": "DET", "OP": "?"}, {"POS": "ADJ", "OP": "*"}, {"POS": "NOUN", "OP": "+"}],
      [{"POS": "ADP"}, {"POS": "DET", "OP": "?"}, {"POS": "ADJ", "OP": "*"}, {"POS": "PRON"}]]

vpMatcher = Matcher(spacyModel.vocab)
vpMatcher.add("VP", VP)
ppMatcher = Matcher(spacyModel.vocab)
ppMatcher.add("PP", PP)

# Extract verb phrases from negative and positive sentences
print("extracting verb and preposition phrases...")
negVP = set()
negPP = set()
posVP = set()
posPP = set()
for sentence in negSentences:
    sentence = spacyModel(sentence)
    vpMatches = vpMatcher(sentence)
    ppMatches = ppMatcher(sentence)
    for match_id, start, end in vpMatches:
        span = sentence[start:end]
        phrasePol = sa.polarity_scores(span.text)
        if phrasePol['compound'] <= -0.5:
            negVP.add(span.text)
    for match_id, start, end in ppMatches:
        span = sentence[start:end]
        phrasePol = sa.polarity_scores(span.text)
        if phrasePol['compound'] <= -0.5:
            negPP.add(span.text)

for sentence in posSentences:
    sentence = spacyModel(sentence)
    vpMatches = vpMatcher(sentence)
    ppMatches = ppMatcher(sentence)
    for match_id, start, end in vpMatches:
        span = sentence[start:end]
        phrasePol = sa.polarity_scores(span.text)
        if phrasePol['compound'] >= 0.5:
            posVP.add(span.text)
    for match_id, start, end in ppMatches:
        span = sentence[start:end]
        phrasePol = sa.polarity_scores(span.text)
        if phrasePol['compound'] >= 0.5:
            posPP.add(span.text)

# print(list(negVP))
# print(list(negPP))

# Create CFG rules w/ specific noun and desired descriptors
print("creating CFG rules...")
grammarString = """
S -> XP VP PP [0.7] | XP VP [0.3]
XP -> Det Nom [0.5] | Nom [0.5]
Nom -> X [0.5] | Adj X [0.5]
V -> 'is' [1]
Det -> 'the' [1]
"""
grammarString += "\nX -> '" + inputWord + "' [1]"
posString = grammarString
negString = grammarString
posString += "\nVP -> V Adj " + " [" + str(1/(len(posVP) + 1)) + "]"
negString += "\nVP -> V Adj " + " [" + str(1/(len(negVP) + 1)) + "]"

for mod in posModifiers:
    posString += "\nAdj -> '" + mod + "'" + " [" + str(1/(len(posModifiers))) + "]"

for mod in negModifiers:
    negString += "\nAdj -> '" + mod + "'" + " [" + str(1/(len(negModifiers))) + "]"

for phrase in list(negVP):
    negString += "\nVP -> '" + phrase + "'" + " [" + str(1/(len(negVP) + 1)) + "]"

for phrase in list(posVP):
    posString += "\nVP -> '" + phrase + "'" + " [" + str(1/(len(posVP) + 1)) + "]"

for phrase in list(negPP):
    negString += "\nPP -> '" + phrase + "'" + " [" + str(1/(len(negPP))) + "]"

for phrase in list(posPP):
    posString += "\nPP -> '" + phrase + "'" + " [" + str(1/(len(posPP))) + "]"

posGrammar = PCFG.fromstring(posString)
negGrammar = PCFG.fromstring(negString)

# Generate sentences w/ CFG rules
print("generating text w/ CFG...")
generatedNeg = list(generate(negGrammar, n=100000))
generatedPos = list(generate(posGrammar, n=100000))
random.shuffle(generatedNeg)
random.shuffle(generatedPos)
scoredNeg = set()
scoredPos = set()
for sentence in generatedNeg:
    sentence = ' '.join(sentence)
    if sa.polarity_scores(sentence)['compound'] <= -0.6:
        scoredNeg.add(sentence)
for sentence in generatedPos:
    sentence = ' '.join(sentence)
    if sa.polarity_scores(sentence)['compound'] >= 0.5:
        scoredPos.add(sentence)
print(list(scoredNeg))

# Pos tag and convert Noun Phrases to CFG?

# Use two embedding matrices to harness descriptors for pos. and neg.? Yes, probably do this.
