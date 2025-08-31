import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import RMSprop

filepath = tf.keras.utils.get_file("shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

text = text[300000:800000]

characters = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))

index_to_char = dict((i, c) for i, c in enumerate(characters))

sequence_length = 40
step_size = 3

sentences = []
next_character = []

for i in range(0, len(text) - sequence_length, step_size):
    sentences.append(text[i: i + sequence_length])
    next_character.append(text[i + sequence_length])

x = np.zeros((len(sentences), sequence_length, len(characters)), dtype=bool)
y = np.zeros((len(sentences), len(characters)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_character[i]]] = 1
    
model = Sequential()
model.add(LSTM(128, input_shape=(sequence_length, len(characters))))
model.add(Dense(len(characters), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))    
model.fit(x, y, batch_size=256, epochs=20)

model.save('poetic_model.h5')




