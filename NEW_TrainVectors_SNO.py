
# coding: utf-8

# In[1]:


import gensim
import tokenization
import os
import collections
import smart_open
import json
import multiprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pprint import pprint

import re

def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return m.group() if m else None

def read_label(fname):
    conceptLabelDict = {}
    errors = []
    with smart_open.smart_open(fname) as f:
        for i, line in enumerate(f):
            # get the id for each concept paragraph
            splitted = line.decode("iso-8859-1").split("\t")
            if len(splitted) == 3:
                conceptID = get_trailing_number(splitted[1])
                conceptLabelDict[conceptID] = splitted[2].replace("\r\n", "")
            else:
                errors.append(splitted)
    return conceptLabelDict, errors

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens
# In[2]:
#global variabls

# directory_path = "/home/h/hl395/mlontology/SNO/"
directory_path = "SNO/"
data_path = directory_path + "data/"
hier = "clinical_finding"
cnn_model_path = directory_path + "cnnModel/"
train_file = data_path + hier + "/ontClassTopology_sno_clinical_finding_small.txt"
vocab_file = "MODEL/small/vocab.txt"

tokenizer = tokenization.FullTokenizer(vocab_file)
label_file_2017 = directory_path + "data/ontClassLabels_july2017.txt"
conceptLabelDict_2017, _ = read_label(label_file_2017)

# training iterations
iterations_list = [10]
for iterations in iterations_list:
    vector_model_path = directory_path + "vectorModel/" + hier + "/" + str(iterations) + "/"
    if (os.path.exists(vector_model_path) == False):
        os.makedirs(vector_model_path)

    doc_ids = []
    def read_corpus(fname, tokens_only=False):
        with smart_open.smart_open(fname) as f:
            for i, line in enumerate(f):
                # get the id for each concept paragraph
                splitted = line.decode("iso-8859-1").split("\t", 1)

                line = splitted[1]
                if tokens_only:
                    yield gensim.utils.simple_preprocess(line)
                else:
                    # For training data, add tags
                    tagID = get_trailing_number(splitted[0])
                    doc_ids.append(tagID)
                    # yield gensim.models.doc2vec.TaggedDocument(tokenizer.tokenize(line), [tagID])
                    yield gensim.models.doc2vec.TaggedDocument(tokenize_text(line), [tagID])
                    # yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [tagID])


    # In[7]:
    train_corpus = list(read_corpus(train_file))

    # In[8]:

    print(len(train_corpus))
    print(train_corpus[1296:1299])

    # In[9]:

    cores = multiprocessing.cpu_count()

    print(cores)
    models = [
        # PV-DBOW
        Doc2Vec(dm=0, dbow_words=1, vector_size=128, window=10, min_count=1, epochs=iterations, workers=cores, alpha=0.025, min_alpha=0.00001, negative=5),
        # PV-DM w/average
        Doc2Vec(dm=1, dm_mean=1, vector_size=128, window=10, min_count=1, epochs=iterations, workers=cores, alpha=0.025, min_alpha=0.00001, negative=5),
    ]



    models[0].build_vocab(train_corpus)
    print(str(models[0]))
    models[1].reset_from(models[0])
    print(str(models[1]))

    # In[11]:

    for model in models:
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        print("one model trained")

    # In[12]:

    for i, model in enumerate(models):
        temp_path = vector_model_path + "model" + str(i)  # creates a temp file
        model.save(str(temp_path))

    # In[13]:
    test_line = "finding of hormone level	measurement finding above reference range	increased androgen level	increased human chorionic gonadotropin level"
    for model in models:
        print(str(model))
        pprint(gensim.utils.simple_preprocess(test_line))
        inferred_vector = model.infer_vector(gensim.utils.simple_preprocess(test_line))
        pprint(model.docvecs.most_similar([inferred_vector], topn=10))

    for model in models:
        ranks = []
        for doc_id in doc_ids:
            # concept_label = tokenizer.tokenize(conceptLabelDict_2017[doc_id])
            # concept_label = gensim.utils.simple_preprocess(conceptLabelDict_2017[doc_id])
            concept_label = tokenize_text(conceptLabelDict_2017[doc_id])
            # print(concept_label)
            inferred_vector = model.infer_vector(concept_label)
            sims = model.docvecs.most_similar([inferred_vector], topn=2000)
            rank = [docid for docid, sim in sims]
            if doc_id in rank:
                rank = rank.index(doc_id)
            else:
                rank = -1
            ranks.append(rank)
        print(collections.Counter(ranks))





