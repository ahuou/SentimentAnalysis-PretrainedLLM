import pickle

import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from data_retriever import data1, data2
import seaborn as sns
import numpy as np

simpleSentiments = data1.loc[:, 'Sentiment']
tweetSentiments = data2.loc[:, 'Sentiment']

def read(path):
    pathWrite = "yob.txt"
    f = open(path, "wb")
    k = pickle.dumps(simpleSentiments)
    f2 = open(path, "wb")
    k = pickle.load(f)
    f.close()
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
    for i in range(3, len(arr)):
        arr[i] = arr[i][1:9]
        if mode == "continuous":
            score = 0.5
            try:
                score = float(arr[i][1:4])
            except:
                score = 0.5

            if score < lowBound and score > 0:
                arr[i] = 'Negative'
            elif score < highBound:
                arr[i] = 'neutral'
            elif score >= highBound and score < 1:
                arr[i] = 'Positive'

        if type == 'tweet':
            print(arr[i])
            print(i)
            if mode=="discrete" and arr[i].lower() == ' neutral' or arr[i].lower() == ' neutal':
                arr[i] = 'neutral'


        missed = True
        for classi in classes:
            if arr[i] in classi:
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



def evaluator(path, type, normalized=False, mode="discrete", lowBound=0.5, highBound=0.5, majMap="Positive"):

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









evaluator("results/DS2_3_Shot_res.txt", "tweet", normalized=True, mode="discrete", lowBound=0.5, highBound=0.5, majMap='Negative')