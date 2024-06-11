import pickle

import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from data_retriever import data1, data2
import seaborn as sns
import numpy as np
from fuzzywuzzy import fuzz
from nltk.corpus import words
from nltk.metrics import edit_distance
import nltk

# Ensure the necessary NLTK resources are available
nltk.download('words')
from nltk.corpus import cmudict
from nltk.metrics import edit_distance

from metaphone import doublemetaphone

def normalize_text(text):
    return ''.join(char.lower() for char in text if char.isalnum())

def classify_by_edit_distance(prediction, categories):
    # Calculate edit distances
    scores = {category: edit_distance(normalize_text(prediction), normalize_text(category)) for category in categories}
    best_fit = min(scores, key=scores.get)
    if scores[best_fit] <= 2:  # Allowing up to 2 edits
        return best_fit
    return None

def classify_by_phonetics(prediction, categories):
    pred_phonetics = doublemetaphone(normalize_text(prediction))
    category_phonetics = {category: doublemetaphone(normalize_text(category)) for category in categories}

    for category, phonetics in category_phonetics.items():
        if pred_phonetics[0] == phonetics[0] or pred_phonetics[1] == phonetics[1]:
            return category
    return None

def classify_sentiment(prediction, categories, majMap):
    methods = [classify_by_edit_distance, classify_by_phonetics]

    for method in methods:
        result = method(prediction, categories)
        if result in categories:
            return (result, 0)

    # Fallback to Levenshtein fuzzy matching if other methods fail
    scores = {category: fuzz.ratio(prediction.lower(), category.lower()) for category in categories}
    best_fit = max(scores, key=scores.get)
    if scores[best_fit] > 80:
        return (best_fit, 0)

    return (majMap, 1)

simpleSentiments = data1.loc[:, 'Sentiment']
tweetSentiments = data2.loc[:, 'Sentiment']

def read(path):
    f2 = open(path, "rb")
    k = pickle.load(f2)
    f2.close()
    print(k)
    return k

def cleaner(path, type, mode="discrete", lowBound=0.5, highBound=0.5, majMap="Positive"):
    if type == "simple":
        new_sentiments = simpleSentiments.copy()
        classes = ["Positive", "Negative"]
    elif type == "tweet":
        classes = ["Positive", "Negative", "neutral"]
        new_sentiments = tweetSentiments.copy()
    else:
        return "Error, type isn't supported"

    arr = read(path)
    count = 0
    for i in range(len(arr)):
        arr[i] = (arr[i][0:9]).strip()
        arr[i], missed = classify_sentiment(arr[i], classes, majMap)

        if mode == "continuous":
            score = 0.5
            try:
                score = float(arr[i][0:4].strip())
            except:
                score = 0.5

            if score < lowBound and score > 0:
                arr[i] = 'Negative'
            elif score < highBound:
                arr[i] = 'neutral'
            elif score >= highBound and score < 1:
                arr[i] = 'Positive'

        if type == 'tweet':
            #print(arr[i])
            #print(i)
            if mode=="discrete" and (arr[i].lower() == 'neutal' or arr[i].lower() == 'neutral'):
                arr[i] = 'neutral'


        for classi in classes:
            if not missed and (arr[i] in classi) :
                arr[i] = classi
                missed = False
                break
        if missed:
            arr[i] = majMap
            count += 1
        #new_sentiments[i] = "Misclassified"


    print("count : ", count)
    print("arr.shape : ", len(arr))
    print("true.shape : ", len(new_sentiments[:len(arr)]))
    print(arr)

    print("test", classify_sentiment("neutrative", classes, majMap))

    return arr, new_sentiments[:len(arr)]


def conf_Matrix(confMatrix, predClasses, trueClasses, normalized):
    sns.set(font_scale=3.0)
    if not normalized:
        sns.heatmap(confMatrix, annot=True, xticklabels=predClasses, yticklabels=trueClasses)
    else:
        cmn = confMatrix.astype('float') / confMatrix.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=predClasses, yticklabels=trueClasses)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    return plt



def evaluator(path, type, normalized=False, mode="discrete", lowBound=0.5, highBound=0.5, majMap="Negative"):

    pred, true = cleaner(path, type, mode=mode, lowBound=lowBound, highBound=highBound, majMap=majMap)
    print("pred : ", len(pred), "\ntrue : ", len(true))

    if type == "simple":
        predClasses = ['Negative', 'Positive']
        trueClasses = ['Negative', 'Positive']

        print(len(true), len(pred))

        confusionMatrix = confusion_matrix(true, pred)

        print("Simple Confusion Matrix: ", confusionMatrix)
        conf_Matrix(confusionMatrix, predClasses, trueClasses, normalized).show()
        simpleClassificationReport = sklearn.metrics.classification_report(true, pred)
        print(simpleClassificationReport)
        return confusionMatrix, simpleClassificationReport


    elif type == "tweet":
        predClasses = ['Negative',  'Positive', 'neutral']
        trueClasses = ['Negative',  'Positive', 'neutral']
        confusionMatrix = confusion_matrix(true, pred)
        #confusionMatrix =  np.array([[796,164,1392], [ 179, 499, 821], [123, 11, 146]])
        print("Tweet Confusion Matrix", confusionMatrix)
        conf_Matrix(confusionMatrix, predClasses, trueClasses, normalized).show()
        tweetClassificationReport = sklearn.metrics.classification_report(true, pred)
        print(tweetClassificationReport)
        return confusionMatrix, tweetClassificationReport

    else:
        return "Error, type not valid"

evaluator("results/DS1_0_Shot_res.txt", "simple", normalized=True, mode="discrete", lowBound=0.5, highBound=0.5, majMap='Negative')