import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re

from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('pos.txt', 'r', encoding='utf-8') as f:
    texts_true = f.readlines()
    texts_true[0] = texts_true[0].replace('\ufeff', '')

with open('neg.txt', 'r', encoding='utf-8') as f:
    texts_false = f.readlines()
    texts_false[0] = texts_false[0].replace('\ufeff', '')

with open('test_texts.txt', 'r', encoding='utf-8') as file:
    otziv = file.readlines()
    otziv[0] = otziv[0].replace('\ufeff', '')




texts = texts_true + texts_false
count_true = len(texts_true)
count_false = len(texts_false)
total_lines = count_true + count_false
#print(count_true, count_false, total_lines)


maxWordsCount = 160
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts(texts)

dist = list(tokenizer.word_counts.items())
#print(dist[:10])
#print(texts[0][:100])


max_text_len = 10
data = tokenizer.texts_to_sequences(texts)
data_pad = pad_sequences(data, maxlen=max_text_len)
#print(data_pad)

#print( list(tokenizer.word_index.items()) )


X = data_pad
Y = np.array([[1, 0]]*count_true + [[0, 1]]*count_false)
#print(X.shape, Y.shape)

indeces = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
X = X[indeces]
Y = Y[indeces]


model = Sequential()
model.add(Embedding(maxWordsCount, 128, input_length = max_text_len))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(0.1))

history = model.fit(X, Y, batch_size=25, epochs=100)

reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

def sequence_to_text(list_of_indices):
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)

data = tokenizer.texts_to_sequences([otziv])
data_pad = pad_sequences(data, maxlen=max_text_len)


res = model.predict(data_pad)
print(res, sep='\n')
if (np.argmax(res) == 1):
    print("негативный(точность указана справа)")
else:
    print("положительный(точность указана слева)")