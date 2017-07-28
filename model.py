import multiprocessing
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense,Dropout
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
import sys
import helper
import random
from gensim.models import Word2Vec
from nltk import word_tokenize
import numpy as np

vec_dim = 300
maxlen = 30
n_iterations = 30  # ideally more, since this improves the quality of the word vecs
n_exposures = 30
window_size = 7
batch_size = 32
nb_epoch = 10
cpu_count = multiprocessing.cpu_count()

# clean the text
text_clean = helper.textclean('description.txt')

# vocabulary of the text
vocab = helper.vocab_build(text_clean)

# creating the vec 2 word model

word_model = Word2Vec(size = vec_dim,
                 min_count = n_exposures,
                 window = window_size,
                 workers = cpu_count,
                 iter = n_iterations)

# training the word model

word_model.build_vocab(text_clean)
word_model.train(text_clean)

# creating mapping dictionaries
word_idx, idx_word, word_vec = helper.ind(vocab,word_model)

# creating embedding matrix
embed_weight = helper.embed_wt(vocab,word_idx,word_vec,vec_dim)

# sequence
sent_seq_parsed,sent_seq_parsed_pad = helper.tokenize(text_clean,word_idx,maxlen)

# training data
word_list = word_tokenize(text_clean)
n_symbols = len(vocab)+1
sentences_list = []

for sentence in sent_seq_parsed_pad:
    for idx in sentence:
        sentences_list.append(idx)
sentences_seq = []
next_word = []

for i in range(len(sentences_list)-maxlen):
    sentences_seq.append(sentences_list[i: i + maxlen])
    next_word.append(sentences_list[i + maxlen])

X = np.zeros((len(sentences_seq), maxlen, n_symbols), dtype=np.bool)
Y = np.zeros((len(sentences_seq), n_symbols), dtype=np.bool)

for i, sent in enumerate(sentences_seq):
    for j, index in enumerate(sent):
        X[i,j,index] = 1
    Y[i,next_word[i]] = 1

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

model = Sequential()
model.add(Embedding(output_dim = vec_dim,
                    input_dim = n_symbols,
                    mask_zero = True,
                    weights = [embed_weight],
                    input_length = maxlen))

model.add(LSTM(vec_dim,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(vec_dim, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(n_symbols, activation = 'softmax'))
model.compile(loss='catergorical_crossentropy',optimizer='adam')

# defining checkpoint
filepath="text_gen-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(train_x,train_y,batch_size=batch_size,nb_epoch=nb_epoch,callbacks=callbacks_list)


# helper function to sample an index from a probability array
def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

random.seed(1113)
x = random.randint(0,test_x.shape[0]-1)
x_test = test_x[x,:,:]
initialized_sent = ''
for i in range(n_symbols):
    initialized_sent += ' '.join(x_test[0, 0, i])

print(initialized_sent)

for i in range(500):
    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, 0.5)
    next_word = idx_word[next_index]
    initialized_sent += ' '.join(next_word)
    del sentence[0]
    sentence.append(next_word)
    sys.stdout.write(' ')
    sys.stdout.write(next_word)
    sys.stdout.flush()
print()