import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.io import wavfile
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment

PATH = './train/train_new/train_'
TEST_PATH = './test/test_new/test_'
def load_speeches(path):
    all_waves = []
    for i in range(90000):
        file = path + str(i) + '.wav'
        if i<90000:
            _, samples = wavfile.read(file)
        else:
            samples = (AudioSegment.from_file(file) +12).get_array_of_samples()
        all_waves.append(samples)
    data = pd.read_csv('train.csv')
    labels = [data.iloc[:, 1][i] for i in range(90000)]
    return all_waves,labels
def load_speeches_test(path):
    all_waves = []
    for i in range(24750):
        file = path + str(i) + '.wav'
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

# def visualize(freqs, spectros):
#     _ , ax = plt.subplots(1, 5, figsize=(10, 2.5), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
#     fig, ax = plt.subplots(10)
#     for i in range(10):
#         ax[i].plot(freqs[i], spectros[i])
#     plt.show()

all_waves,labels = load_speeches(PATH)
# mapping = {21: 0, 31: 1, 41: 2, 32: 3, 42: 4, 43: 5}
# labels = [mapping[int(label)] for label in labels]
labelencoder = LabelEncoder().fit(labels)
encoded_labels = tf.keras.utils.to_categorical(labelencoder.transform(labels), 6)

# max_sequence_len = max([len(x) for x in all_waves])
all_waves = np.array(pad_sequences(all_waves, maxlen=6000, dtype='int16', padding='post'))
freqs,tims,spectros = get_spectrograms(all_waves)
spectros = np.array(spectros) #spectros[0].shape --> (129, 26)
spectros = spectros.reshape(120000, 129, 26, 1)
X, X_test, Y, Y_test = train_test_split(spectros, encoded_labels, test_size=0.15, random_state=42)




#https://www.kaggle.com/rkuo2000/spoken-digit-mnist
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (5,5), activation='relu',padding='same', input_shape=(129, 26, 1)),
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
  tf.keras.layers.Dense(6, activation='softmax')
])

#https://www.kaggle.com/vassylkorzh/speech-recognition-lb-0-74
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(8, 2, activation='relu',padding='valid', input_shape=(129, 26, 1)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(8, 2, activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(8, 2, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Conv2D(16, 3, activation='relu',padding='same'),
  tf.keras.layers.AveragePooling2D((2, 2)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#https://github.com/guptajay/Kaggle-Digit-Recognizer/blob/master/Digit_Recognizer_MNIST.ipynb
model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(filters = 32, kernel_size = 5, strides = 1, activation = 'relu', input_shape = (129,26,1), kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
tf.keras.layers.Conv2D(filters = 32, kernel_size = 5, strides = 1, use_bias=False),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Activation('relu'),
tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2),
tf.keras.layers.Dropout(0.25),
tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, use_bias=False),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Activation('relu'),
tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2),
tf.keras.layers.Dropout(0.25),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(units = 256, use_bias=False),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Activation('relu'),
tf.keras.layers.Dense(units = 128, use_bias=False),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Activation('relu'),
tf.keras.layers.Dense(units = 84, use_bias=False),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Activation('relu'),
tf.keras.layers.Dropout(0.25),
tf.keras.layers.Dense(units = 6, activation = 'softmax')
])


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu',padding='same', input_shape=(129, 26, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (1,1), activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.002)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.002)),
  tf.keras.layers.Dense(6, activation='softmax')
])

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu',padding='same', input_shape=(129, 26, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (2,2), activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.002)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (1,1), activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.002)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.002)),
  tf.keras.layers.Dense(6, activation='softmax')
])



model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X,Y,batch_size=128,epochs=10,validation_data=(X_test,Y_test))

#model.save('./model')
#new_model = tf.keras.models.load_model('./model')
#model.predict(X_test)
def plot_model(model):
    history = model.history
    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(len(acc))
    plt.figure(figsize=(10,7))
    plt.plot(epochs, acc, 'r')
    plt.plot(epochs, val_acc, 'b')
    plt.title('Training and validation accuracy')
    plt.legend(['Training accuracy','Validation accuracy'])
    plt.figure()
    plt.figure(figsize=(10,7))
    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.title('Training and validation loss')
    plt.legend(['Training loss','Validation loss'])
    plt.figure()

test_waves = load_speeches_test(TEST_PATH)
_, _, test_spectros = get_spectrograms(test_waves)
test_spectros = np.array(test_spectros)
test_spectros = test_spectros.reshape(24750, 129, 26, 1)
predictions = model.predict(test_spectros)
# decoded_predictions = np.argwhere(predictions ==1 )#[:, 1]
inverse_predictions = np.array([ np.array([i, np.argmax(prediction)]) for i, prediction in enumerate(predictions)])
inverse_predictions[:, 1] = labelencoder.inverse_transform(inverse_predictions[:, 1]) #len(np.argwhere(inverse_predictions[:, 1] == 43))
# for replace in replacements: inverse_predictions[replace][1] = 43
df = pd.DataFrame(inverse_predictions, columns=['ID', 'Label'])
df.to_csv('sub.csv', index=False)

"""
#load predictions and find intersection of 43
intersection = np.argwhere(all_predictions[:, 0] ==43).flatten()
for i in range(1,26):
    pred = np.argwhere(all_predictions[:, i] ==43).flatten()
    intersection = np.intersect1d(intersection, pred)
np.savetxt('./intersection.txt', intersection)
"""

"""
#load predictions and analyze statistics

all_predictions = np.loadtxt("./predictions/prediction0.txt").astype(int).reshape(24750, 1) #shape:(24750, 50)
for i in range(1,26):
     next_prediction = np.loadtxt(f"./predictions/prediction{i}.txt").astype(int).reshape(24750,1)
     all_predictions = np.append(all_predictions, next_prediction, axis=1)

#check where predictions aren't all the same
modes = stats.mode(all_predictions, axis=1)[0].flatten() #shape:(24750, 1)

find_discrepancies = np.array([np.all(mode == row_prediction) for mode, row_prediction in zip(modes, all_predictions)])
discrepancies = np.argwhere(find_discrepancies==False)

percentage_equals_mode = np.array( [ ((all_predictions[i] == modes[i]).sum() /51) for i in range(24570) ] )
less_than98 = np.argwhere(percentage_equals_mode<0.98).shape
less_than70 = np.argwhere(percentage_equals_mode<0.7)


array = np.array(range(24750)).reshape(24750,1)
inverse_predictions = np.append(array, modes.reshape(24750, 1), axis=1)
"""


"""
#generate 43 files
for i in range(100):
    predictions = model.predict(test_spectros)
    inverse_predictions = np.array([ np.array([i, np.argmax(prediction)]) for i, prediction in enumerate(predictions)])
    inverse_predictions[:, 1] = labelencoder.inverse_transform(inverse_predictions[:, 1])
    np.savetxt(f"./data/43_predictions{i}.txt", np.argwhere(inverse_predictions[:, 1] == 43))
    model.fit(X,Y,batch_size=128,epochs=1,validation_data=(X_test,Y_test))

#save predictions
i = 0
while i<51:
    if model.history.history['val_accuracy'][0] < 0.9999:
        print('skipped, too low')
        model.fit(X,Y,batch_size=128,epochs=1,validation_data=(X_test,Y_test))
        continue
    predictions = model.predict(test_spectros)
    inverse_predictions = np.array([ np.array([i, np.argmax(prediction)]) for i, prediction in enumerate(predictions)])
    inverse_predictions[:, 1] = labelencoder.inverse_transform(inverse_predictions[:, 1])
    np.savetxt(f"./predictions/prediction{i}.txt", inverse_predictions[:, 1])
    print(f'iteration {i}')
    model.fit(X,Y,batch_size=128,epochs=1,validation_data=(X_test,Y_test))
    i+=1



#cummulative intersection
intersection = np.loadtxt("./data/43_predictions100.txt").astype(int)
for i in range(101, 130):
    pred =  np.loadtxt(f"./data/43_predictions{i}.txt").astype(int)
    intersection = np.intersect1d(intersection, pred)
np.savetxt('./intersection.txt', intersection)

"""

"""
import os
import time
import random
import librosa
import numpy as np
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


PATH = './train/train_new/train_'
def extract_mfcc(path):
    ft_batch = []
    for i in range(120000):
        file = path + str(i) + '.wav'
        if i<90000:
            _, samples = wavfile.read(file)
        else:
            samples = (AudioSegment.from_file(file) +12).get_array_of_samples()
            samples = np.pad(samples, (0, 6000-len(samples)), 'constant' )
        # raw_w, sampling_rate = librosa.load(file, sr=None)
        mfcc_features = librosa.feature.mfcc(samples, sampling_rate)
        ft_batch.append(mfcc_features)
    return ft_batch

def extract_mfcc_test(path):
    ft_batch = []
    for i in range(24750):
        file = path + str(i) + '.wav'
        raw_w, sampling_rate = librosa.load(file, sr=None)
        mfcc_features = librosa.feature.mfcc(raw_w, sampling_rate)
        ft_batch.append(mfcc_features)
    return ft_batch

def display_power_spectrum(mfcc):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.show()
def get_labels():
    data = pd.read_csv('train.csv')
    labels = [data.iloc[:, 1][i] for i in range(len(data))]
    return labels

ft_batch = extract_mfcc(PATH)
ft_batch = np.array(ft_batch)
labels = get_labels()
# encoder = LabelEncoder()
# labels = encoder.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels, 43)

spectros = ft_batch.reshape(90000, 20, 12, 1)

X, X_test, Y, Y_test = train_test_split(spectros, labels, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (5,5), activation='relu',padding='same', input_shape=(20, 12, 1)),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
  tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dense(43, activation='softmax')
])


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu',padding='same', input_shape=(20, 12, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (1,1), activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.002)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.002)),
  tf.keras.layers.Dense(43, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X,Y,batch_size=32,epochs=10,validation_data=(X_test,Y_test))



test_fr = extract_mfcc_test(TEST_PATH)
test_fr = np.array(test_fr)
test_spectros = test_fr.reshape(24750, 20, 12, 1)
predictions = model.predict(test_spectros)
decoded_predictions = np.array([ np.array([i, np.argmax(prediction)]) for i, prediction in enumerate(predictions)])
df = pd.DataFrame(decoded_predictions, columns=['ID', 'Label'])

import pywt
def get_wavelets(all_waves):
    wavelets = []
    for wave in all_waves:
        coeff, freq = pywt.dwt(wave, 'db1')
        wavelets.append([coeff, freq])
    return np.array(wavelets)
def get_fft(all_waves):
    fft_array = []
    for wave in all_waves:
        fft_array.append(np.abs(np.fft.fft(wave)))
    return np.array(fft_array)

"""



"""
one_sample, one_wave = wavfile.read('1_nicolas_24.wav')
three_sample, three_wave = wavfile.read('3_george_32.wav')
one_wave = np.pad(one_wave, (0, 6000-len(one_wave)), 'constant' )
three_wave = np.pad(three_wave, (0, 6000-len(three_wave)), 'constant' )

test_sample, test_wave = wavfile.read('test_0.wav')

sound1 = (AudioSegment.from_file('1_nicolas_24.wav')) + 12
sound3 = (AudioSegment.from_file('3_george_32.wav')) + 12
combined = sound1.overlay(sound3)
comb_modified = combined + 2
plt.plot(combined.get_array_of_samples())
plt.plot(comb_modified.get_array_of_samples())
plt.plot(test_wave)



"""

"""
def load_speeches(path):
    all_waves = []
    for i in range(18000):
        file = path + str(i) + '.wav'
        _, samples = wavfile.read(file)
        all_waves.append(samples)
    data = pd.read_csv('train.csv')
    labels = [data.iloc[:, 1][i] for i in range(18000)]
    return all_waves,labels
def append_43(all_waves, labels, intersection):
    for i in intersection:
        file = TEST_PATH + str(i) + '.wav'
        _, samples = wavfile.read(file)
        all_waves.append(samples)
        labels.append(43)
    return all_waves, labels

all_waves,labels = load_speeches(PATH)
all_waves, labels = append_43(all_waves, labels, intersection)
labelencoder = LabelEncoder().fit(labels)
encoded_labels = tf.keras.utils.to_categorical(labelencoder.transform(labels), 6)

freqs,tims,spectros = get_spectrograms(all_waves)
spectros = np.array(spectros) #spectros[0].shape --> (129, 26)
spectros = spectros.reshape(len(all_waves), 129, 26, 1)
X, X_test, Y, Y_test = train_test_split(spectros, encoded_labels, test_size=0.15, random_state=98)

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X,Y,batch_size=128,epochs=71,validation_data=(X_test,Y_test))
"""
