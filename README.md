# Topic-based Document-Level Sentiment Analysis using Contextual Cues


Packages needed:
- pandas
- numpy
- ntlk
- sklearn
- spacy
- glove_python
- gensim
- tensorflow
- keras
- stop_words
- matplotlib
- seaborn
- scipy

### Clean the text, do topic modeling, extract word embeddings and topic2vec emebdings
- use file FILE_INPUT with columns ['review', 'polarity']
- save the clean dataset into FILE_OUTPUT
- builds the topic models
- creates the word embeddings and topic2vec into the models/ folder

```
python3.7 clean_data.py FILE_INPUT FILE_OUTPUT
```

### Create the Doc2Vec and DocTopic2Vec models
- uses de models in the the models/ folder
- requires the clean file (the output from previous step) as a parameter EMB_NAME
- requires the embedding name as a parameter EMB_NAME 


```
python3.7 create_doc2vec.py CLEAN_DATASET EMB_NAME
```


### Classification
- For classification, there are 4 different files
- All of them use as an input a file with the Doc2Vec/DocTopic2Vec embeddings
- The DNN classification also requires as input the Doc2Vec/DocTopic2Vec dimensions (i.e., VEC_DIM)
- For more information regarding the file names and embedding size see filenames.txt

```
# Logistic Regression
python3.7 classification_logreg.py INPUT_FILE

 VEC_DIM

# RNN DNNs
python3.7 classification_rnn.py INPUT_FILE VEC_DIM

# CNN DNNs
python3.7 classification_cnn.py INPUT_FILE VEC_DIM

# CNN DNNs that do not work on the a single gpu
python3.7 classification_multigpu.py INPUT_FILE VEC_DIM
```

We wish you Good Luck and God's speed running this ;)
