# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:45:29 2018

@author: Tugba
"""
import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
#import keras.backend as K


def buildModel(preCalculatedWeightPath=None, input_size = 32, num_classes = 6):
    #Now we can go ahead and create our Convolution model
    model = Sequential()
     
    model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',
                  input_shape=(input_size, input_size, 1), data_format="channels_last"))

    model.add(MaxPooling2D((2, 2)))
    #model.add(Dropout(0.25))
    model.add(Conv2D(64, (7, 7), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    
    print ("Create model successfully")
    if preCalculatedWeightPath:
        model.load_weights(preCalculatedWeightPath)
    
    model.compile(loss=keras.losses.categorical_crossentropy, 
                      optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    
    return model





