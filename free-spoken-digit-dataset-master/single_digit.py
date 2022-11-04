import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from os.path import isdir, join
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.io import wavfile
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D

data_path = './recordings2'
TEST_PATH = './test/test_new/test_'

def load_speeches(path):
    waves = [f for f in os.listdir(path) if f.endswith('.wav')]
    labels = []
    samples_rate = []
    all_waves = []
    for wav in waves:
        sample_rate, samples = wavfile.read(join(path,wav))
        samples_rate.append(sample_rate)
        labels.append(wav[0])
        all_waves.append(samples)
    return all_waves ,samples_rate,labels
def load_speeches_test(PATH):
    all_waves = []
    for i in range(24750):
        file = PATH + str(i) + '.wav'
        _, samples = wavfile.read(file)
        all_waves.append(samples)
    return all_waves

def get_spectrograms(waves):
    sample_rate = 8000
    spectros = []
    freqs = []
    tims = []
    for wav in waves:
        frequencies, times, spectrogram = signal.spectrogram(wav, sample_rate)
        freqs.append(frequencies)
        tims.append(times)
        spectros.append(spectrogram)
    return freqs,tims,spectros

all_waves,samples_rate,labels = load_speeches(data_path) #len(labels) = 3000
# max_sequence_len = max([len(x) for x in all_waves])
all_waves = np.array(pad_sequences(all_waves, maxlen=6000, padding='post'))
freqs,tims,spectros = get_spectrograms(all_waves)
spectros = np.array(spectros)
spectros = spectros.reshape(1500,129,26,1)
labels = tf.keras.utils.to_categorical(labels, 5)
X, X_test, Y, Y_test = train_test_split(spectros, labels, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (5,5), activation='relu',padding='same', input_shape=(129, 26,1)),
  tf.keras.layers.Conv2D(32,(5,5), activation='relu',padding='same'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Dropout((0.25)),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2)),
  tf.keras.layers.Dropout((0.25)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dropout((0.5)),
  tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X,Y,batch_size=16,epochs=100,validation_data=(X_test,Y_test))


test_waves = load_speeches_test(TEST_PATH)
_, _, test_spectros = get_spectrograms(test_waves)
test_spectros = np.array(test_spectros)
test_spectros = test_spectros.reshape(24750, 129, 26, 1)
predictions = model.predict(test_spectros)


decoded_predictions = np.array([ np.array([i, np.argmax(prediction)]) for i, prediction in enumerate(predictions)])
df = pd.DataFrame(decoded_predictions, columns=['ID', 'Label'])
df.to_csv('sub.csv', index=False)
