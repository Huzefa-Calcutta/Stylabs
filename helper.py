import os,sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import sent_tokenize,word_tokenize
from keras.preprocessing.sequence import pad_sequences
import re

def textclean(path):
    des = open(path,'rt').read().replace('\\n','.').replace('..','.')
    des = re.sub('[^a-zA-Z0-9 .,?]','',des)
    return des

def vocab_build(text):
    vocab = list(set(word_tokenize(text)))
    return vocab

def ind(vocab, model):
    w2ind = {vocab[i]:i+1 for i in range(len(vocab))}
    w2ind['0'] = ''
    ind2w = {v:k for k,v in w2ind.iteritems()}
    w2vec = {word: model[word] for word in w2ind.keys()}
    return w2ind,ind2w, w2vec

def tokenize(text, w2ind, max_len):
    sentence_list = sent_tokenize(text)
    parsed_sent_list = []
    for i in range(len(sentence_list)):
        sentence = sentence_list[i]
        parsed_sent = [w2ind[k] for k in word_tokenize(sentence)]
        parsed_sent_list.append(parsed_sent)
    parsed_sent_list_pad = pad_sequences(parsed_sent_list,max_len)
    return parsed_sent_list, parsed_sent_list_pad

def embed_wt(vocab,w2ind,w2vec,vec_dim = 200):
    n_symbols = len(vocab) + 1
    embedding_weights = np.zeros((n_symbols, vec_dim))
    for word in w2vec.keys():
        embedding_weights[w2ind[word], :] = w2vec[word]
    return embedding_weights



