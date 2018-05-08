# -*- coding:utf-8 -*-
'''
    VGG-16 model
'''

from __future__ import print_function

import keras
import data_process
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

batch_size = 128
num_classes = 2
epochs = 30
img_rows, img_cols = 68, 1688
(x_train, y_train), (x_test, y_test) = data_process.load_data('../data/fetal_image_01_30_25K.npy')

model_vgg = VGG16(weights=None, include_top=True, input_shape=(img_rows, img_cols, 1), classes=2)
model_vgg.add(Dense(num_classes, activation='softmax', name='predictions'))
model_vgg.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model_vgg.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2,
                    shuffle=True,
                    validation_data=(x_test, y_test))

model_vgg.save('../trained_model/vgg.h5')
print("test set")
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
