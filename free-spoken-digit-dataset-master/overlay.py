from pydub import AudioSegment
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy import signal
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split

PATH = './recordings'
ones = [str for str in os.listdir(PATH) if str.startswith('1')]
twos = [str for str in os.listdir(PATH) if str.startswith('2')]
threes = [str for str in os.listdir(PATH) if str.startswith('3')]
fours = [str for str in os.listdir(PATH) if str.startswith('4')]

one_two = [(one,two) for one in ones for two in twos]
one_three = [(one,three) for one in ones for three in threes]
one_four = [(one,four) for one in ones for four in fours]

two_three = [(two, three) for two in twos for three in threes]
two_four = [(two, four) for two in twos for four in fours]

three_four = [(three, four) for three in threes for four in fours]



def extract_waves(array):
    result = []
    for i in range(90000):
        sound1 = AudioSegment.from_file(PATH + '/' + array[i][0]) + 20
        sound2 = AudioSegment.from_file(PATH + '/' + array[i][1]) + 20
        combined = sound1.overlay(sound2)
        result.append( combined.get_array_of_samples() )
    result = np.array(pad_sequences(result, maxlen=6000, padding='post'))
    return result

one_two_waves = extract_waves(one_two)
one_three_waves = extract_waves(one_three)
one_four_waves = extract_waves(one_four)
two_three_waves = extract_waves(two_three)
two_four_waves = extract_waves(two_four)
three_four_waves = extract_waves(three_four)

def get_spectrograms(waves):
    sample_rate = 8000
    spectros = []
    for wav in waves:
        _, _, spectrogram = signal.spectrogram(wav, sample_rate)
        spectros.append(spectrogram)
    return np.array(spectros)

one_two_spectros = get_spectrograms(one_two_waves)
one_three_spectros = get_spectrograms(one_three_waves)
one_four_spectros = get_spectrograms(one_four_waves)
two_three_spectros = get_spectrograms(two_three_waves)
two_four_spectros = get_spectrograms(two_four_waves)
three_four_spectros = get_spectrograms(three_four_waves)

spectros1 = np.append(one_two_spectros, one_three_spectros, axis = 0)
spectros2 = np.append(one_four_spectros, two_three_spectros, axis = 0)
spectros3 = np.append(two_four_spectros, three_four_spectros, axis = 0)

spectros = np.append(spectros1, spectros2, axis = 0)
spectros = np.append(spectros, spectros3, axis = 0)
spectros = spectros.reshape(540000, 129, 26, 1)

labels = np.append( np.append( np.repeat(21, 90000), np.repeat(31, 90000)), np.append( np.repeat(41, 90000), np.repeat(32, 90000)) )
labels = np.append(labels, np.append( np.repeat(42, 90000), np.repeat(43, 90000)))


labelencoder = LabelEncoder().fit(labels)
encoded_labels = tf.keras.utils.to_categorical(labelencoder.transform(labels), 6)

# X, X_test, Y, Y_test = train_test_split(spectros, encoded_labels, test_size=0.2, random_state=42)
# X_part = X[:144000, :, :, :]
# X_test_part = X_test[:36000, :, :, :]
# Y_part = Y[:144000, :]
# Y_test_part = Y_test[:36000, :]

# X_part = X[144000:288000, :, :, :]
# X_test_part = X_test[36000:72000, :, :, :]
# Y_part = Y[144000:288000, :]
# Y_test_part = Y_test[36000:72000, :]

X_part = X[288000:, :, :, :]
X_test_part = X_test[72000:, :, :, :]
Y_part = Y[288000:, :]
Y_test_part = Y_test[72000:, :]
"""
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu',padding='same', input_shape=(129, 26, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (1,1), activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.002)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.002)),
  tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_part,Y_part,batch_size=512,epochs=40,validation_data=(X_test_part,Y_test_part))

#model.save('./model')
"""


"""

"""
