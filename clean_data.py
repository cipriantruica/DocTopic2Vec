# coding: utf-8

__author__      = "Ciprian-Octavian Truică, Elena-Simona Apostol, Maria-Luiza Șerban"
__copyright__   = "Copyright 2021, University Politehnica of Bucharest"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "{ciprian.truica,elena.apostol,maria_luiza.serban}@upb.ro"
__status__      = "Development"

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from tokenization import Tokenization
from gensim import models, corpora
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
# Plotting tools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Word2Vec
from gensim.models import Word2Vec

# FastText
from gensim.models import FastText

# Glove
from glove import Corpus, Glove
from scipy.sparse import coo_matrix

import os
import sys

no_threads = cpu_count()
tkn = Tokenization()

NUM_FEATURES = 15
NUM_ITER = 10
NUM_TOPICS = 10

MODELS_FOLDER = './models/'


W2V_CBOW_MODEL_FILE_256 ='W2V_CBOW_256.model'
W2V_CBOW_MODEL_FILE_100 ='W2V_CBOW_100.model'
W2V_SG_MODEL_FILE_128 = 'W2V_SG_100.model'
W2V_SG_MODEL_FILE_100 = 'W2V_SG_128.model'
FT_CBOW_MODEL_FILE_256 = 'FT_CBOW_256.model'
FT_CBOW_MODEL_FILE_100 = 'FT_CBOW_100.model'
FT_SG_MODEL_FILE_128 = 'FT_SG_128.model'
FT_SG_MODEL_FILE_100 = 'FT_SG_100.model'
GLOVE_MODEL_FILE_128 = 'GLOVE_128.model'
GLOVE_MODEL_FILE_100 = 'GLOVE_100.model'


def processElement(row):
    clean_text = tkn.createCorpus(str(row[0]))
    if len(clean_text) > 0:
        return [row[0], clean_text, row[1]]
    else:
        return None

def normalizeDataSet(dataSet):
    clean_texts = []
    with ProcessPoolExecutor(max_workers=no_threads) as worker:
        for result in worker.map(processElement, dataSet.to_numpy()):
            if result:
                clean_texts.append(result)
    return clean_texts


def stringToList(text):
    resultList = text.split("', '")
    resultList[0] = resultList[0][2:]
    resultList[-1] = resultList[-1][:-2]
    return resultList

def updateNormaliseColumn(dataSet):
    stringList = []
    for row in dataSet:
        stringList.append(stringToList(row))
    return stringList

def trainW2V(corpus, n_dim, n_epochs, window_size, n_alpha, t_sg):
    #w2vModel = Word2Vec(x_train, size=n_dim, workers = cores, window = window_size, alpha = n_alpha, sg = t_sg)
    w2vModel = Word2Vec(size=n_dim, workers = no_threads, window = window_size, alpha = n_alpha, sg = t_sg)
    w2vModel.build_vocab(corpus)#[x.words for x in x_train]
    w2vModel.train(corpus, total_examples = w2vModel.corpus_count, epochs = n_epochs)#[x.words for x in x_train]
    return w2vModel

def trainFastText(corpus, n_dim, n_epochs, window_size, n_alpha, t_sg):
    fastTextModel = FastText(size=n_dim, workers = no_threads, window = window_size, alpha = n_alpha, sg = t_sg)
    fastTextModel.build_vocab(corpus)
    fastTextModel.train(corpus, total_examples = fastTextModel.corpus_count, epochs = n_epochs)
    return fastTextModel

def trainGlove(n_dim, n_epochs, window_size, n_alpha):
    corpus = Corpus()  
    #corpus.fit(dataSet['normalized'].sample(n=40000), window = window_size)
    corpus.fit(dataSet['normalized'], window = window_size)
    #gloveModel = Glove (no_components = n_dim, learning_rate = n_learning_rate, alpha = n_alpha)
    gloveModel = Glove(no_components = n_dim, alpha = n_alpha)
    gloveModel.fit(corpus.matrix, epochs = n_epochs, no_threads = no_threads, verbose = True)
    gloveModel.add_dictionary(corpus.dictionary)
    return corpus, gloveModel

def getVectorizedData(data):
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1, lowercase=False)
    vectorizer_fit = vectorizer.fit(data)
    data_vectorized = vectorizer_fit.transform(data)
    return data_vectorized, vectorizer, vectorizer_fit

def printTopics(model, vectorizer, top_n=NUM_FEATURES):# top n features for each topics (list)
    df_topic_keywords = pd.DataFrame(columns=['topic', 'word', 'importance'])
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]])
        for i in topic.argsort()[:-top_n - 1:-1]:
            df_topic_keywords.append({'topic': idx, 'word': vectorizer.get_feature_names()[i], 'importance': topic[i]}, ignore_index=True)
    print(df_topic_keywords)
    return df_topic_keywords

def showKeywords(name, vectorizer, model, n_words=NUM_FEATURES):# top n keywords for each topic (dataframe)
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    df_topic_keywords = pd.DataFrame(topic_keywords)  # Topic - Keywords Dataframe
    df_topic_keywords.columns = ['Word '+ str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    df_topic_keywords.to_csv(MODELS_FOLDER + name + '_topic_keywords.csv', encoding = "utf-8", index = False, header=True)
    print(df_topic_keywords.to_string())

def evaluateModel (name, model, output, vectorizer, data_vectorized):
    print(name + " (no_docs, no_topics) = ", output.shape, "\n") 
    print(model, "\n")
    topic_keywords_importance = printTopics(model, vectorizer)
    print()
    showKeywords(name, vectorizer, model)  
    df_document_topic, df_topic_distribution =  showTopics(model, output, data_vectorized)
    df_topic_distribution.to_csv(MODELS_FOLDER +  name + '_document_topics.csv', encoding = "utf-8", index = False, header=True)
    print(df_document_topic.head(15).to_string())
    print()
    print(df_topic_distribution)
    return df_document_topic

def showTopics(model, output, data_vectorized):# dominant topic in each document
    # Create Document - Topic Matrix
    #output = best_model.transform(data_vectorized)
    topic_names = ["Topic" + str(i) for i in range(model.n_components)]# column names
    doc_names = [str(i) for i in range(data_vectorized.shape[0])]# index names
    df_document_topic = pd.DataFrame(np.round(output, 2), columns=topic_names, index=doc_names)
    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic
    # topics distribution across documents
    df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
    df_topic_distribution.columns = ['Topic Num', 'Num Documents']
    return df_document_topic, df_topic_distribution


def topicsDataframe(model, vectorizer, df_topic_keywords, top_n=NUM_FEATURES):
    for idx, topic in enumerate(model.components_):
        for i in topic.argsort()[:-top_n - 1:-1]:
            df_topic_keywords = df_topic_keywords.append({'topic': "Topic% d" % (idx), 'word': vectorizer.get_feature_names()[i], 'importance': topic[i]}, ignore_index=True)
    return df_topic_keywords

def buildTopicVector(model, df, size):
    # average review
    vec = np.zeros(size).reshape(1, size)
    count = 0.
    for i in df.index:
        word = df['word'][i]
        try:
            if(isinstance(model, Glove)):
                current = model.word_vectors[model.dictionary[word]]
            else:
                current = model[word]
            vec += current * df['importance'][i]
            count += 1.
        except KeyError: 
            continue
    if count != 0:
        vec /= count
    return vec[0]

def topic2Vec(model, df_topic_keywords, df_topic_vec, size):
    for topic in df_topic_keywords.topic.unique():
        vec = buildTopicVector(model, df_topic_keywords.loc[df_topic_keywords.topic == topic], size)
        df_topic_vec = df_topic_vec.append({'topic': topic.split()[1], 'vec': vec}, ignore_index=True)
    return df_topic_vec

if __name__ =="__main__":
    FILE_INPUT = sys.argv[1]
    FILE_OUTPUT = sys.argv[2]

    dataSet = pd.read_csv(FILE_INPUT, encoding = "utf-8")
    dataSet.loc[dataSet['sentiment'] == 'positive', 'sentiment'] = 2
    dataSet.loc[dataSet['sentiment'] == 'neutral', 'sentiment'] = 1
    dataSet.loc[dataSet['sentiment'] == 'negative', 'sentiment'] = 0

    normalizedDataSet = normalizeDataSet(dataSet)

    # if uploaded from file and not processed
    # dataSet.normalized = dataSet.normalized.apply(stringToList)
    dataSet = pd.DataFrame(normalizedDataSet, columns=['review', 'normalized', 'polarity'])
    dataSet = dataSet[dataSet.normalized.isnull() == False]

    

    n_epochs = 30
    window_size = 4
    n_alpha = 1e-02
    n_dim_100 = 100
    n_dim_128 = 128
    n_dim_256 = 256

    t_sg = 0
    w2vCBOWModel_256 = trainW2V(dataSet['normalized'], n_dim_256, n_epochs, window_size, n_alpha, 0)
    w2vCBOWModel_100 = trainW2V(dataSet['normalized'], n_dim_100, n_epochs, window_size, n_alpha, t_sg)
    
    w2vCBOWModel_256.save(MODELS_FOLDER + W2V_CBOW_MODEL_FILE_256)
    w2vCBOWModel_100.save(MODELS_FOLDER + W2V_CBOW_MODEL_FILE_100)

    t_sg = 1

    w2vSGModel_128 = trainW2V(dataSet['normalized'], n_dim_128, n_epochs, window_size, n_alpha, t_sg)
    w2vSGModel_100 = trainW2V(dataSet['normalized'], n_dim_100, n_epochs, window_size, n_alpha, t_sg)

    w2vSGModel_128.save(MODELS_FOLDER + W2V_SG_MODEL_FILE_128)
    w2vSGModel_100.save(MODELS_FOLDER + W2V_SG_MODEL_FILE_100)

    t_sg = 0

    fastTextCBOWModel_256 = trainFastText(dataSet['normalized'], n_dim_256, n_epochs, window_size, n_alpha, t_sg)
    fastTextCBOWModel_100 = trainFastText(dataSet['normalized'], n_dim_100, n_epochs, window_size, n_alpha, t_sg)

    fastTextCBOWModel_256.save(MODELS_FOLDER + FT_CBOW_MODEL_FILE_256)
    fastTextCBOWModel_100.save(MODELS_FOLDER + FT_CBOW_MODEL_FILE_100)

    t_sg = 1

    fastTextSGModel_128 = trainFastText(dataSet['normalized'], n_dim_128, n_epochs, window_size, n_alpha, t_sg)
    fastTextSGModel_100 = trainFastText(dataSet['normalized'], n_dim_100, n_epochs, window_size, n_alpha, t_sg)

    fastTextSGModel_128.save(MODELS_FOLDER + FT_SG_MODEL_FILE_128)
    fastTextSGModel_100.save(MODELS_FOLDER + FT_SG_MODEL_FILE_100)

    corpus_128, gloveModel_128 = trainGlove(n_dim_128, n_epochs, window_size, n_alpha)
    corpus_100, gloveModel_100 = trainGlove(n_dim_100, n_epochs, window_size, n_alpha)

    gloveModel_128.save(MODELS_FOLDER + GLOVE_MODEL_FILE_128)
    gloveModel_100.save(MODELS_FOLDER + GLOVE_MODEL_FILE_100)


    data_vectorized, vectorizer, vectorizer_fit = getVectorizedData(dataSet['normalized'])
    tfidf = pd.DataFrame(data=data_vectorized.todense(), columns=vectorizer.get_feature_names())
    TFIDF_FILE = 'TFIDF_MATRIX.csv'
    tfidf.to_csv(MODELS_FOLDER + TFIDF_FILE, encoding = "utf-8", index = False, header=True)

    # Latent Dirichlet Allocation Model
    lda_model = LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                              evaluate_every=-1, learning_decay=0.9,
                              learning_method='online', learning_offset=10.0,
                              max_doc_update_iter=100, max_iter=NUM_ITER,
                              mean_change_tol=0.001, n_components=10, n_jobs=-1,
                              perp_tol=0.1, random_state=100, topic_word_prior=None,
                              total_samples=1000000.0, verbose=1)
    lda_output = lda_model.fit_transform(data_vectorized)

    df_document_topic = evaluateModel("LDA", lda_model, lda_output, vectorizer, data_vectorized)
    dataSet['LDA'] = df_document_topic['dominant_topic'].tolist()


    # Non-Negative Matrix Factorization Model
    nmf_model = NMF(n_components=NUM_TOPICS, max_iter=NUM_ITER)
    nmf_output = nmf_model.fit_transform(data_vectorized)

    df_document_topic = evaluateModel("NMF", nmf_model, nmf_output, vectorizer, data_vectorized)
    dataSet['NMF'] = df_document_topic['dominant_topic'].tolist()

    # Latent Semantic Indexing Model
    lsi_model = TruncatedSVD(n_components=NUM_TOPICS, n_iter=NUM_ITER)
    lsi_output = lsi_model.fit_transform(data_vectorized)

    df_document_topic = evaluateModel("LSI", lsi_model, lsi_output, vectorizer, data_vectorized)
    dataSet['LSI'] = df_document_topic['dominant_topic'].tolist()


    for model in [('LDA', lda_model), ('NMF', nmf_model), ('LSI', lsi_model)]:
        for embedding in [("W2V_CBOW_100", w2vCBOWModel_100, n_dim_100), ("W2V_CBOW_256", w2vCBOWModel_256, n_dim_256), ("W2V_SG_100", w2vSGModel_100, n_dim_100), ("W2V_SG_128", w2vSGModel_128, n_dim_128), ("FT_CBOW_256", fastTextCBOWModel_256, n_dim_256), ("FT_CBOW_100", fastTextCBOWModel_100, n_dim_100), ("FT_SG_128", fastTextSGModel_128, n_dim_128), ("FT_SG_100", fastTextSGModel_100, n_dim_100), ("GLOVE_128", gloveModel_128, n_dim_128), ("GLOVE_100", gloveModel_100, n_dim_100)]:
            df_topic_keywords = pd.DataFrame(columns=['topic', 'word', 'importance'])
            df_topic_vec = pd.DataFrame(columns=['topic', 'vec'])
            df_topic_keywords = topicsDataframe(model[1], vectorizer, df_topic_keywords)
            df_topic_vec = topic2Vec(embedding[1], df_topic_keywords, df_topic_vec, embedding[2])
            print(df_topic_vec)
            filename = model[0] + "_" + embedding[0] + ".csv"
            df_topic_vec.to_csv(MODELS_FOLDER + filename, encoding = "utf-8", index = False, header=True)
    
    dataSet.to_csv(MODELS_FOLDER + FILE_OUTPUT, encoding = "utf-8", index = False, header=True)
            
