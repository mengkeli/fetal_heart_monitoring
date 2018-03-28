# -*- coding:utf-8 -*-
'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
import data_process
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

batch_size = 128
num_classes = 2
epochs = 10

# input image dimensions
img_rows, img_cols = 120, 2402

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = data_process.load_data(data_process.image_file)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

'''step1 : 
'''
model = Sequential()

'''step 2 : 构建网络层
'''

model.add(Conv2D(filters=3, kernel_size=(10, 20),
                 strides=(4, 8),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(5, (10, 10), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
# model.add(Dropout(0.25))
model.add(Flatten())

# model.add(Dense(128, activation='relu')) # 隐藏节点128个
# model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax')) # 最后一层，输出结果是2个类别

'''
step 3 : 编译
可以设置SGD优化函数
'''
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

'''
step 4 : 训练
batch_size：对总的样本数进行分组，每组包含的样本数量
epochs ：训练次数
shuffle：是否把数据随机打乱之后再进行训练
validation_split：拿出百分之多少用来做交叉验证
verbose：屏显模式 0：不输出  1：输出进度  2：输出每次的训练结果
'''
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          shuffle=True,
          validation_data=(x_test, y_test))

model.summary()

'''
step 5 : 输出
'''
print("test set")
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

