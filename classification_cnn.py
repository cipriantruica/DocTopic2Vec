# coding: utf-8

__author__      = "Ciprian-Octavian Truică, Elena-Simona Apostol"
__copyright__   = "Copyright 2021, University Politehnica of Bucharest"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "{ciprian.truica,elena.apostol}@upb.ro"
__status__      = "Development"

import os
import sys

# helpers
import pandas as pd 
import time

# classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
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


#  CNN
def prepareCNNModel1(no_attributes, filters, kernel_size):
    model = Sequential(name = 'CNN')
    model.add(Reshape((no_attributes, 1)))
    model.add(Conv1D(filters = filters, kernel_size=kernel_size, name = 'CNN1'))
    model.add(MaxPooling1D(name='MaxPolling1'))
    model.add(Flatten(name='Flatten1'))
    model.add(Dense(units = 3, activation = 'softmax', name = 'Output'))#sigmoid #softmax
    model.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

#  CNN GRU
def prepareCNNGRUModel1(no_attributes, filters, kernel_size):
    model = Sequential(name = 'CNN-GRU')  
    model.add(Reshape((no_attributes, 1)))
    model.add(Conv1D(filters = filters, kernel_size=kernel_size, name = 'CNN1'))
    model.add(MaxPooling1D(name='MaxPolling1'))
    model.add(GRU(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'GRU1'))
    model.add(Dense(units = 3, activation = 'softmax', name = 'Output'))#sigmoid #softmax
    model.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

# CNN GRU x3
def prepareCNNGRUModel2(no_attributes, filters, kernel_size):
    model = Sequential(name = 'CNN-GRU3')
    model.add(Reshape((no_attributes, 1)))
    model.add(Conv1D(filters = filters, kernel_size=kernel_size, name = 'CNN1'))
    model.add(MaxPooling1D(name='MaxPolling1'))
    model.add(GRU(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'GRU1', return_sequences=True))
    model.add(GRU(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'GRU2', return_sequences=True))
    model.add(GRU(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'GRU3'))
    model.add(Dense(3, activation = 'softmax', name = 'Output'))#sigmoid #softmax
    model.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

#  CNN BiGRU
def prepareCNNBiGRUModel1(no_attributes, filters, kernel_size):
    model = Sequential(name = 'CNN-BiGRU')  
    model.add(Reshape((no_attributes, 1)))
    model.add(Conv1D(filters = filters, kernel_size=kernel_size, name = 'CNN1'))
    model.add(MaxPooling1D(name='MaxPolling1'))
    model.add(Bidirectional(GRU(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'BiGRU1')))
    model.add(Dense(units = 3, activation = 'softmax', name = 'Output'))#sigmoid #softmax
    model.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

# CNN BiGRU x3
def prepareCNNBiGRUModel2(no_attributes, filters, kernel_size):
    model = Sequential(name = 'CNN-BiGRU3')
    model.add(Reshape((no_attributes, 1)))
    model.add(Conv1D(filters = filters, kernel_size=kernel_size, name = 'CNN1'))
    model.add(MaxPooling1D(name='MaxPolling1'))
    model.add(Bidirectional(GRU(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'BiGRU1', return_sequences=True)))
    model.add(Bidirectional(GRU(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'BiGRU2', return_sequences=True)))
    model.add(Bidirectional(GRU(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'BiGRU3')))
    model.add(Dense(3, activation = 'softmax', name = 'Output'))#sigmoid #softmax
    model.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

#  CNN LSTM
def prepareCNNLSTMModel1(no_attributes, filters, kernel_size):
    model = Sequential(name = 'CNN-LSTM')  
    model.add(Reshape((no_attributes, 1)))
    model.add(Conv1D(filters = filters, kernel_size=kernel_size, name = 'CNN1'))
    model.add(MaxPooling1D(name='MaxPolling1'))
    model.add(LSTM(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'LSTM1'))
    model.add(Dense(units = 3, activation = 'softmax', name = 'Output'))#sigmoid #softmax
    model.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

# CNN LSTM x3
def prepareCNNLSTMModel2(no_attributes, filters, kernel_size):
    model = Sequential(name = 'CNN-LSTM3')
    model.add(Reshape((no_attributes, 1)))
    model.add(Conv1D(filters = filters, kernel_size=kernel_size, name = 'CNN1'))
    model.add(MaxPooling1D(name='MaxPolling1'))
    model.add(LSTM(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'LSTM1', return_sequences=True))
    model.add(LSTM(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'LSTM2', return_sequences=True))
    model.add(LSTM(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'LSTM3'))
    model.add(Dense(3, activation = 'softmax', name = 'Output'))#sigmoid #softmax
    model.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

#  CNN BiLSTM
def prepareCNNBiLSTMModel1(no_attributes, filters, kernel_size):
    model = Sequential(name = 'CNN-BiLSTM')  
    model.add(Reshape((no_attributes, 1)))
    model.add(Conv1D(filters = filters, kernel_size=kernel_size, name = 'CNN1'))
    model.add(MaxPooling1D(name='MaxPolling1'))
    model.add(Bidirectional(LSTM(units = 128, dropout = 0.2, recurrent_dropout = 0.2), name = 'BiLSTM1'))
    model.add(Dense(units = 3, activation = 'softmax', name = 'Output'))#sigmoid #softmax
    model.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

# CNN BiLSTM x3
def prepareCNNBiLSTMModel2(no_attributes, filters, kernel_size):
    model = Sequential(name = 'CNN-BiLSTM3')
    model.add(Reshape((no_attributes, 1)))
    model.add(Conv1D(filters = filters, kernel_size=kernel_size, name = 'CNN1'))
    model.add(MaxPooling1D(name='MaxPolling1'))
    model.add(Bidirectional(LSTM(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'BiLSTM1', return_sequences=True)))
    model.add(Bidirectional(LSTM(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'BiLSTM2', return_sequences=True)))
    model.add(Bidirectional(LSTM(units = 128, dropout = 0.2, recurrent_dropout = 0.2, name = 'BiLSTM3')))
    model.add(Dense(3, activation = 'softmax', name = 'Output'))#sigmoid #softmax
    model.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

# MDPI Model
def prepareMDPIModel1(no_attributes, filters, kernel_size):
    model = Sequential(name = 'MDPI-Model')  
    model.add(Reshape((no_attributes, 1)))
    model.add(Conv1D(filters = filters, kernel_size=5, activation='relu', name = 'CNN1'))
    model.add(MaxPooling1D(name='MaxPolling1'))
    model.add(Conv1D(filters = filters, kernel_size=4, activation='relu', name = 'CNN2'))
    model.add(MaxPooling1D(name='MaxPolling2'))
    model.add(Conv1D(filters = filters, kernel_size=3, activation='relu', name = 'CNN3'))
    model.add(MaxPooling1D(name='MaxPolling3'))
    model.add(Conv1D(filters = filters, kernel_size=2, activation='relu', name = 'CNN4'))
    model.add(MaxPooling1D(name='MaxPolling4'))
    model.add(Bidirectional(LSTM(units = 128, dropout = 0.2, recurrent_dropout = 0.2), name = 'BiLSTM1'))
    model.add(Dense(units = 3, activation = 'softmax', name = 'Output'))#sigmoid #softmax
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

    x_train_np = np.array(x_train)
    x_test_np = np.array(x_test)

    no_attributes = x_test_np.shape[1]
    filters = 64
    kernel_size = int(no_attributes/2)

    # CNN
    trainNN(x_train_np, x_test_np, y_vec_train, y_vec_test, epochs_n, prepareCNNModel1, no_attributes, filters, kernel_size)
    # CNN GRU
    trainNN(x_train_np, x_test_np, y_vec_train, y_vec_test, epochs_n, prepareCNNGRUModel1, no_attributes, filters, kernel_size)
    # CNN GRU3
    trainNN(x_train_np, x_test_np, y_vec_train, y_vec_test, epochs_n, prepareCNNGRUModel2, no_attributes, filters, kernel_size)
    #  CNN BiGRU
    trainNN(x_train_np, x_test_np, y_vec_train, y_vec_test, epochs_n, prepareCNNBiGRUModel1, no_attributes, filters, kernel_size)
    #  CNN BiGRU3
    trainNN(x_train_np, x_test_np, y_vec_train, y_vec_test, epochs_n, prepareCNNBiGRUModel2, no_attributes, filters, kernel_size)
    # CNN LSTM
    trainNN(x_train_np, x_test_np, y_vec_train, y_vec_test, epochs_n, prepareCNNLSTMModel1, no_attributes, filters, kernel_size) 
    # CNN LSTM3
    trainNN(x_train_np, x_test_np, y_vec_train, y_vec_test, epochs_n, prepareCNNLSTMModel2, no_attributes, filters, kernel_size)
    # CNN BiLSTM
    trainNN(x_train_np, x_test_np, y_vec_train, y_vec_test, epochs_n, prepareCNNBiLSTMModel1, no_attributes, filters, kernel_size)   
    # CNN BiLSTM3
    trainNN(x_train_np, x_test_np, y_vec_train, y_vec_test, epochs_n, prepareCNNBiLSTMModel2, no_attributes, filters, kernel_size)
    # MDPI
    trainNN(x_train_np, x_test_np, y_vec_train, y_vec_test, epochs_n, prepareMDPIModel1, no_attributes, filters, kernel_size)





