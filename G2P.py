import numpy as np
import tqdm
from tqdm import tqdm_notebook as tqdm
import math
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Masking, Dense
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop
from keras.layers import Bidirectional, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint

text = open('train.txt', 'r')
input_data = [[], []]
g2p_dict, p2g_dict = {}, {}
for lines in text:
    line = lines.split()
    grapheme = line[0]
    for phoneme in line[1:]:
        g2p_dict.update({grapheme: phoneme})
        p2g_dict.update({phoneme: grapheme})
        input_data[0].append(grapheme)
        input_data[1].append(phoneme)


# input_data = np.array(input_data)

def makeVocabularySet(data, sep=False):
    vocab = {}
    sequences = []
    max_seq_len = 0
    i = 0
    for rows in data:
        if sep:
            rows = rows.split('_')
        if len(rows) > max_seq_len:
            max_seq_len = len(rows)
        for c in rows:
            if c in vocab.keys():
                continue
            vocab.update({c: i})
            i += 1
    rev_vocab = dict((v, k) for (k, v) in vocab.items())
    return max_seq_len, vocab, rev_vocab


graph_max_seq_len, grapheme_encoder, grapheme_decoder = makeVocabularySet(input_data[0])
phone_max_seq_len, phoneme_encoder, phoneme_decoder = makeVocabularySet(input_data[1], True)


# Не заупскай этот код дважды
def add_token(vocab, rev_vocab, token):
    n = len(vocab)
    vocab.update({token: n})
    rev_vocab.update({n: token})


# add_token(phoneme_encoder, phoneme_decoder, '<go>')
add_token(phoneme_encoder, phoneme_decoder, '<end>')
add_token(grapheme_encoder, grapheme_decoder, '<end>')

num_grapheme = len(grapheme_encoder)
num_phoneme = len(phoneme_encoder)
graphemes = input_data[0]
phonemes = input_data[1]


def encode_sequence(data, vocab, split=False):
    encoded = []
    for rows in data:
        if split:
            # rows = '<go>_' + rows   #  add go-token (for phonemes only)
            rows = rows.split('_')
        tmp = list(map(lambda x: vocab[x], rows))
        tmp.append(vocab['<end>'])  # add end-token
        encoded.append(tmp)
    return np.array(encoded)


encoded_graphemes = encode_sequence(graphemes, grapheme_encoder)
graph_max_seq_len += 1
encoded_phonemes = encode_sequence(phonemes, phoneme_encoder, True)
phone_max_seq_len += 2


def padding(data, vocab):
    padded = []
    max_len = max(graph_max_seq_len, phone_max_seq_len)
    for row in data:
        add = [vocab['<end>'] for i in range(max_len - len(row))]
        padded.append(row + add)
    return padded


padded_graphemes = padding(encoded_graphemes, grapheme_encoder)
padded_phonemes = padding(encoded_phonemes, phoneme_encoder)


def vectorization(data, vocab):
    shp = [len(data), len(data[0]), len(vocab)]
    train_data = np.zeros((shp[0], shp[1], shp[2]), dtype=np.int)
    for i in range(shp[0]):
        j = 0
        for k in data[i]:
            train_data[i][j][k] = 1
            j += 1
    i += 1
    return train_data


X_train = vectorization(padded_graphemes, grapheme_encoder)
y_train = vectorization(padded_phonemes, phoneme_encoder)

outs = y_train.shape[2]
outs1 = y_train.shape[1]
max_len, feats = X_train.shape[1], X_train.shape[2]
hidden_l = 128
batch_size = 64
epochs = 500

model = Sequential()
model.add(Masking(mask_value=0., input_shape=(max_len, feats)))
# model.add(Bidirectional(LSTM(hidden_l, recurrent_dropout=0.01, return_sequences=True,
#                              activation='tanh', recurrent_activation='hard_sigmoid'), merge_mode="sum",
#                         input_shape=(max_len, feats)))
model.add(LSTM(hidden_l, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid'))
model.add(
    LSTM(hidden_l, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', go_backwards=True))
model.add(GRU(hidden_l, return_sequences=True, recurrent_activation='hard_sigmoid'))
model.add(GRU(hidden_l, return_sequences=True, recurrent_activation='hard_sigmoid', go_backwards=True))
model.add(TimeDistributed(Dense(256, activation='relu')))
model.add(TimeDistributed(Dense(outs, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer="rmsprop")

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2,
          callbacks=[EarlyStopping(min_delta=0.0001, verbose=0, mode='auto', patience=10)])


def word_to_vector(word):
    encoded = encode_sequence(word, grapheme_encoder)
    padded = padding(encoded, grapheme_encoder)
    return vectorization(padded, grapheme_encoder)


words = ['GREEN', 'CONFERENCE', 'ELEVEN']
x = word_to_vector(words)
pred = model.predict(x, verbose=1)
decode = lambda x: phoneme_decoder[np.argmax(x)]
print('_'.join(map(decode, pred[0])))

test = pd.read_csv('test.csv')
test = list(test['Word'])

print(test[:5])
x = word_to_vector(test)
print(x.shape)

y = model.predict(x, verbose=1)


def decode(pred):
    result = []
    for pr in pred:
        s = ''
        for phon in pr:
            c = phoneme_decoder[np.argmax(phon)]
            if c != '<go>' and c != '<end>':
                s += c + '_'
        result.append(s[:-1])
    return result


prediction = decode(y)

print(prediction[:4])


def saver(path, res):
    with open(path, 'w')as out:
        print('Id,Transcription', file=out)
        for i in tqdm(range(len(res))):
            print('{num},{res}'.format(num=i + 1, res=str(res[i])), file=out)


saver('result.csv', prediction)
print(len(prediction))
