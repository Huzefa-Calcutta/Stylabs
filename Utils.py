import os,sys
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
from nltk import sent_tokenize


#Reading the description file
des = open("description.txt",'rt').read().replace('\\n','.')
des = re.sub('[^a-zA-Z0-9 .]','',des)
des_list = sent_tokenize(des)
vocab = set(des.replace('.',' ').split())
def word2ind(vocab):
    wtoind = {}
    for e in vocab
        