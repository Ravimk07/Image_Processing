# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 12:17:52 2018

@author: admin
"""

import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

# load data into train and test

(X_train, y_train),(X_test, y_test) = mnist.load_data()

# Preprocess input data

X_train = X_train.reshape(X_train.shape[0], 28,28,1)
X_test = X_test.reshape(X_test.shape[0], 28,28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
 
# 6. Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Define model Architecture

model = Sequential()

model.add(Convolution2D(32,3,3, activation='relu', input_shape=(28,28,1)))
model.add(Convolution2D(32,3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile model

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# fit model on training data

model.fit(X_train, Y_train,
          batch_size=32, nb_epoch=10,verbose=1)

#Evaluate model on test data

score = model.evaluate(X_test, Y_test, verbose=0)



