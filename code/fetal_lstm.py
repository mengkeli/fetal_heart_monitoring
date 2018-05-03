# -*- coding:utf-8 -*-
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Activation, Dense, Embedding
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
import data_process

(x_train, y_train), (x_test, y_test) = data_process.load_data(data_process.series_smooth_file)

# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

neurons = 512
activation_function = 'tanh'
loss = 'mse'
optimizer="adam"
dropout = 0.25
batch_size = 12
epochs = 53
window_len = 7
training_size = 0.8
output_size = 1


model = Sequential()
model.add(LSTM(neurons, return_sequences=True, input_shape=(inputs.shape[1], inputs.shape[2]), activation=activation_function))
model.add(Dropout(dropout))
model.add(LSTM(neurons, return_sequences=True, activation=activation_function))
model.add(Dropout(dropout))
model.add(LSTM(neurons, activation=activation_function))
model.add(Dropout(dropout))
model.add(Dense(units=output_size))
model.add(Activation(activation_function))
model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
model.summary()

# fit network
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=2, shuffle=False)

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.savefig('../data/lstm_history.png',bbox_inches='tight', edgecolor='white')
plt.show()



