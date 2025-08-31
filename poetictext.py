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

'''

for i in range(0, len(text) - sequence_length, step_size):
    sentences.append(text[i: i + sequence_length])
    next_character.append(text[i + sequence_length])

x = np.zeros((len(sentences), sequence_length, len(characters)), dtype=bool)
y = np.zeros((len(sentences), len(characters)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_character[i]]] = 1
    
'''
    
model = tf.keras.models.load_model('poetry_model.h5')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - sequence_length - 1)
    generated = ''
    sentence = text[start_index: start_index + sequence_length]
    generated += sentence
    print('Generating with seed: "' + sentence + '"')
    for i in range(length):
        x_pred = np.zeros((1, sequence_length, len(characters)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
    return generated


print("--------0.2--------")
print(generate_text(400, 0.2))
print("--------0.4--------")
print(generate_text(400, 0.4))
print("--------0.6--------")
print(generate_text(400, 0.6))
print("--------0.8--------")
print(generate_text(400, 0.8))
print("--------1--------")
print(generate_text(400, 1.0))








