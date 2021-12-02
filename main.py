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
nltk.download('punkt')
nltk.download('vader_lexicon')


# Adds noun to dictionary with frequency and document source index
def addNounToDict(noun, dictionary, indexSource):
    if noun in dictionary:
        dictionary[noun][0] = dictionary[noun][0] + 1
        dictionary[noun][1].add(indexSource)
    else:
        dictionary[noun] = [1, set()]
        dictionary[noun][1].add(indexSource)


# Generates text from CFG by randomly picking a rule at every depth (until a terminal)
def randomGenerate(grammar, rule):
    sentence = ''
    if rule in grammar._lhs_index:
        subRules = grammar._lhs_index[rule]
        index = random.randint(0, len(subRules) - 1)
        for sub in subRules[index]._rhs:
            sentence += randomGenerate(grammar, sub)
    else:
        sentence += " " + rule
    return sentence


# Load in dependency parser
print("loading pos tagger and dependency parser...")
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
                    addNounToDict(token.text.lower(), nouns, len(allSentences) - 1)
        for ne in sentenceDoc.ents:
            addNounToDict(ne.text.lower(), namedEntities, index)

# Sort noun dictionary by frequency
print("sorting noun dictionaries...")
descNouns = {key: val for key, val in sorted(nouns.items(), key=lambda element: element[1], reverse=True)}
descNE = {key: val for key, val in sorted(namedEntities.items(), key=lambda element: element[1], reverse=True)}

# Get top ~25 nouns and entities referenced and filter out punct.
print("getting top nouns/entities...")
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
        if inputWord in mostRefNouns:
            nounInstances = list(descNouns[inputWord][1])
            break
        elif inputWord in mostRefNE:
            nounInstances = list(descNE[inputWord][1])
            break
        else:
            print("Please pick a word from either list.")
    else:
        print("Please pick a word from either list.")

# Get user choice for desired sentiment
inputSen = ""
genPos = False
while True:
    inputSen = input("Do you want to generate negative or positive sentiment about this entity? Enter 'n' or 'p': ")
    if type(inputSen) is str:
        inputSen = inputSen.lower()
        if inputSen == 'p':
            genPos = True
            break
        elif inputSen == 'n':
            genPos = False
            break
        else:
            print("Please enter 'p' for positive or 'n' for negative: ")
    else:
        print("Please enter 'p' for positive or 'n' for negative: ")

# Use nltk vader sentiment analysis to get instances of neg/pos sentences w/ modifiers
print("running sa and retrieving pos/neg modifiers...")
sa = SentimentIntensityAnalyzer()
allSentences = list(allSentences)
negSentences = set()
posSentences = set()
negModifiers = set()
posModifiers = set()
for sentence in allSentences:
    sentencePolarity = sa.polarity_scores(sentence)
    if sentencePolarity['compound'] >= 0.5:
        if genPos:
            posSentences.add(sentence)
            sentenceDep = spacyModel(sentence)
            for token in sentenceDep:
                if token.dep_ == "amod":
                    posModifiers.add(token.text.lower())
    elif sentencePolarity['compound'] <= -0.5:
        if not genPos:
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
if not genPos:
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
else:
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

# Create CFG rules w/ specific noun and desired descriptors
print("creating CFG rules...")
grammarString = """
S -> XP VP PP | XP VP
XP -> X | Adj X
V -> 'is'
VP -> V Adj
"""
grammarString += "\nX -> '" + inputWord + "'"
posString = grammarString
negString = grammarString

if genPos:
    for mod in posModifiers:
        posString += "\nAdj -> '" + mod + "'"
    for phrase in list(posVP):
        posString += "\nVP -> '" + phrase + "'"
    for phrase in list(posPP):
        posString += "\nPP -> '" + phrase + "'"
    posGrammar = CFG.fromstring(posString)
else:
    for mod in negModifiers:
        negString += "\nAdj -> '" + mod + "'"
    for phrase in list(negVP):
        negString += "\nVP -> '" + phrase + "'"
    for phrase in list(negPP):
        negString += "\nPP -> '" + phrase + "'"
    negGrammar = CFG.fromstring(negString)

# Generate sentences w/ CFG rules
print("generating text w/ CFG...")
if genPos:
    generatedPos = []
    for x in range(1000):
        generated = randomGenerate(posGrammar, posGrammar.start())[1:]
        generatedPos.append(generated)
    random.shuffle(generatedPos)
    scoredPos = set()
    for sentence in generatedPos:
        if sa.polarity_scores(sentence)['compound'] >= 0.5:
            scoredPos.add(sentence)
else:
    generatedNeg = []
    for x in range(1000):
        generated = randomGenerate(negGrammar, negGrammar.start())[1:]
        generatedNeg.append(generated)
    random.shuffle(generatedNeg)
    scoredNeg = set()
    for sentence in generatedNeg:
        if sa.polarity_scores(sentence)['compound'] <= -0.6:
            scoredNeg.add(sentence)

# Write generated text to 'GENERATED.txt' file
with open("GENERATED.txt", mode='w') as output:
    if genPos:
        genList = scoredPos
    else:
        genList = scoredNeg
    for sentence in genList:
        output.write(sentence + "\n")
