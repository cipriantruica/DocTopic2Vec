# coding: utf-8

__author__      = "Ciprian-Octavian Truică, Elena-Simona Apostol, Maria-Luiza Șerban"
__copyright__   = "Copyright 2021, University Politehnica of Bucharest"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "{ciprian.truica,elena.apostol,maria_luiza.serban}@upb.ro"
__status__      = "Development"


import os
import sys

# helpers
import pandas as pd 
import time


# classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sn

# split data set
import numpy as np 
from sklearn.model_selection import train_test_split

import multiprocessing as mp
# import keras
import math
from multiprocessing import cpu_count

no_threads = cpu_count() - 2


def stringToStringFloat(text):
    resultList = [float(elem) for elem in text[1:-1].split(",")]
    if len(resultList) != n_dim:
        print('Not OK')
        resultList = [0] * n_dim
        print(len(resultList))
    return resultList

def splitDataSet(dataSet):
    X = dataSet.vec.to_list()
    Y = dataSet.polarity.to_list()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify = Y, test_size=0.2)# keep proportions
    return x_train, x_test, y_train, y_test

def logisticReg(x_train, y_train, x_test):
    # classification 
    start_time = time.time()
    logreg = LogisticRegression(n_jobs=no_threads, C=1e5)
    logreg = logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    print("Time taken to predict: " + str(time.time() - start_time))
    return y_pred

def printReport(y_test, y_pred):
    #score = logreg.score(x_test, y_test)
    accuracyScore = accuracy_score(y_test, y_pred)
    # confMatrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)#, target_names=['0','1','-1'])
    print('accuracy %s' % accuracyScore)
    print(report)
    # print(confusion_matrix)

# def plotConfusionMatrix(y_test, y_pred, modelName):
#     labels = [-1, 0, 1]
#     array = confusion_matrix(y_test, y_pred)
#     df_cm = pd.DataFrame(array, index = [i for i in labels], columns = [i for i in labels])
#     plt.figure(figsize = (10,7))
#     heat_map = sn.heatmap(df_cm, annot=True, fmt='d', square = True, cmap = 'BuPu')
#     plt.xlabel("Predicted outputs")
#     plt.ylabel("Actual outputs")
#     plt.title(modelName + " Confusion Matrix")
#     plt.show()
  
def showResult(y_test, y_pred, modelName):            
    printReport(y_test, y_pred)
    # plotConfusionMatrix(y_test, y_pred, modelName)

if __name__ =="__main__":
    FILE_NAME = sys.argv[1]
    n_dim = int(sys.argv[2])

    dataSet = pd.read_csv(FILE_NAME, encoding = "utf-8")
    dataSet.vec = dataSet.vec.apply(stringToStringFloat)

    # Split Data
    x_train, x_test, y_train, y_test = splitDataSet(dataSet)
        
    #### LogReg
    y_pred = logisticReg(x_train, y_train, x_test)
    showResult(y_test, y_pred, FILE_NAME) 
    