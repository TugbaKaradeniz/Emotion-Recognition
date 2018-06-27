# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:45:29 2018

@author: Tugba
"""
# Used Keras libraries to create model
import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

#import keras.backend as K
from keras import backend as K
K.set_image_dim_ordering('th')

def buildModel(preCalculatedWeightPath=None, input_size = 48):
    
    model = Sequential()

    model.add(ZeroPadding2D ((3, 3), input_shape=(1, 48, 48)))
    model.add(Conv2D (64, (7, 7), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3), strides=(2,2)))

    model.add(ZeroPadding2D((1,1), input_shape=(1, 48, 48)))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add (ZeroPadding2D ((1, 1)))
    model.add (Conv2D (64, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    # 20% dropout
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    
    print ("Create model successfully")
    if preCalculatedWeightPath:
        model.load_weights(preCalculatedWeightPath)
    
    model.compile(loss=keras.losses.categorical_crossentropy, 
                      optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    
    return model





