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
# from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sn

# split data set
import numpy as np 
from sklearn.model_selection import train_test_split

import multiprocessing as mp

# import keras
from keras.models import Sequential,load_model
from keras.layers import Dense, Embedding, GRU, Dropout, LSTM, Bidirectional
# from keras.layers import Attention, Concatenate, Input
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.utils import plot_model, np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import math
import tensorflow as tf
from keras.layers import Conv1D, Flatten, MaxPooling1D, Reshape


# GRU
def prepareGRUModel1():
  model = Sequential(name = 'GRU')
  model.add(GRU(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'GRU1'))
  model.add(Dense(3, activation = 'softmax', name = 'Output'))#sigmoid #softmax
  model.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
  return model
 
#  BiGRU
def prepareGRUModel2():
  model = Sequential(name = 'BiGRU')
  model.add(Bidirectional(GRU(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'BiGRU1')))
  model.add(Dense(3, activation = 'softmax', name = 'Output'))#sigmoid #softmax
  model.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
  return model 

# GRU x3
def prepareGRUModel3():
  model = Sequential(name = 'GRU3')
  model.add(GRU(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'GRU1', return_sequences=True))
  model.add(GRU(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'GRU2', return_sequences=True))
  model.add(GRU(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'GRU3'))
  model.add(Dense(3, activation = 'softmax', name = 'Output'))#sigmoid #softmax
  model.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
  return model

# BiGRU x3
def prepareGRUModel4():
  model = Sequential(name = 'BiGRU3')
  model.add(Bidirectional(GRU(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'BiGRU1', return_sequences=True)))
  model.add(Bidirectional(GRU(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'BiGRU2', return_sequences=True)))
  model.add(Bidirectional(GRU(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'BiGRU3')))
  model.add(Dense(3, activation = 'softmax', name = 'Output'))#sigmoid #softmax
  model.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
  return model

# LSTM
def prepareLSTMModel1():
  model = Sequential(name = 'LSTM')
  model.add(LSTM(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'LSTM1'))
  model.add(Dense(3, activation = 'softmax', name = 'Output'))#sigmoid #softmax
  model.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
  return model

# BiLSTM
def prepareLSTMModel2():
  model = Sequential(name = 'BiLSTM')
  model.add(Bidirectional(LSTM(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'BiLSTM1')))
  model.add(Dense(3, activation = 'softmax', name = 'Output'))#sigmoid #softmax
  model.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
  return model

# LSTM x3
def prepareLSTMModel3():
  model = Sequential(name = 'LSTM3')
  model.add(LSTM(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'LSTM1', return_sequences=True))
  model.add(LSTM(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'LSTM2', return_sequences=True))
  model.add(LSTM(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'LSTM3'))
  model.add(Dense(3, activation = 'softmax', name = 'Output'))#sigmoid #softmax
  model.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
  return model

# BiLSTM x3
def prepareLSTMModel4():
  model = Sequential(name = 'BiLSTM3')
  model.add(Bidirectional(LSTM(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'BiLSTM1', return_sequences=True)))
  model.add(Bidirectional(LSTM(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'BiLSTM2', return_sequences=True)))
  model.add(Bidirectional(LSTM(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'BiLSTM3')))
  model.add(Dense(3, activation = 'softmax', name = 'Output'))#sigmoid #softmax
  model.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
  return model

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

def prepareTrainTestNN(x_train, x_test, y_train, y_test):
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    # convert array of labeled data (from 0 to nb_classes-1) to one-hot vector
    y_train = np_utils.to_categorical(y_train, num_classes=3)
    y_test = np_utils.to_categorical(y_test, num_classes=3)
    return x_train, x_test, y_train, y_test

def evaluate(y_test, y_pred, modelName='GRU', weights=True):
    y_pred_norm = []

    for elem in y_pred:
        line = [ 0 ] * len(elem)
        try:
            # if an error appears here
            # get a random class
            line[elem.tolist().index(max(elem.tolist()))] = 1
        except:
            print("Error for getting predicted class")
            line[rnd.randint(0, len(elem))] = 1
        y_pred_norm.append(line)

    y_p = np.argmax(np.array(y_pred_norm), 1)
    y_t = np.argmax(np.array(y_test), 1)
    accuracyScore = accuracy_score(y_t, y_p)
    # confMatrix = confusion_matrix(y_t, y_p)
    report = classification_report(y_t, y_p)
    print(modelName, ' Accuracy ', accuracyScore)
    print(report)
    

def printNNReport(x_train, y_train, x_test, y_test, model, history):
    print(model.summary())
    # print()
    # loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
    # print("Training Accuracy: {:.4f}".format(accuracy))
    # loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    # print("Testing Accuracy:  {:.4f}".format(accuracy))
    y_pred = model.predict(x_test, verbose=False)
    evaluate(y_test, y_pred, modelName=model.name)

def trainNN(x_train, x_test, y_train, y_test, epochs_n, prepareModel, no_attributes, filters, kernel_size):
    model = prepareModel(no_attributes, filters, kernel_size)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    # mc = ModelCheckpoint(fileName, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    start_time = time.time()
    history = model.fit(x_train, y_train,
                    epochs=epochs_n, 
                    verbose=False,
                    validation_data=(x_test, y_test),
                    batch_size=5000,
                    callbacks=[es])
    print("Time taken to train: " + str(time.time() - start_time))
    printNNReport(x_train, y_train, x_test, y_test, model, history)
    return history, model

if __name__ =="__main__":
    FILE_NAME = sys.argv[1]
    n_dim = int(sys.argv[2])

    dataSet = pd.read_csv(FILE_NAME, encoding = "utf-8")
    dataSet.vec = dataSet.vec.apply(stringToStringFloat)

    # Split Data
    x_train, x_test, y_train, y_test = splitDataSet(dataSet)
    # Create Dataset for NN
    epochs_n = 200
    x_vec_train, x_vec_test, y_vec_train, y_vec_test = prepareTrainTestNN(np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test))


    #### GRU    
    history, model = trainNN(x_vec_train, x_vec_test, y_vec_train, y_vec_test, epochs_n, prepareGRUModel1)

    #### BiGRU
    history, model = trainNN(x_vec_train, x_vec_test, y_vec_train, y_vec_test, epochs_n, prepareGRUModel2)

    #### GRU3
    history, model = trainNN(x_vec_train, x_vec_test, y_vec_train, y_vec_test, epochs_n, prepareGRUModel3)

    #### BiGRU3
    history, model = trainNN(x_vec_train, x_vec_test, y_vec_train, y_vec_test, epochs_n, prepareGRUModel4)

    #### LSTM
    history, model = trainNN(x_vec_train, x_vec_test, y_vec_train, y_vec_test, epochs_n, prepareLSTMModel1)

    #### BiLSTM
    history, model = trainNN(x_vec_train, x_vec_test, y_vec_train, y_vec_test, epochs_n, prepareLSTMModel2)

    #### LSTM3
    history, model = trainNN(x_vec_train, x_vec_test, y_vec_train, y_vec_test, epochs_n, prepareLSTMModel3)

    #### BiLSTM3
    history, model = trainNN(x_vec_train, x_vec_test, y_vec_train, y_vec_test, epochs_n, prepareLSTMModel4)