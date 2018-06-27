#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 23:17:48 2018

@author: Tugba
"""
import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
#from keras.utils import np_utils as u
#import matplotlib.pyplot as plt
#from load_data import make_sets

def builModel(input_size, num_classes, preCalculatedWeightPath=None):
    #Now we can go ahead and create our Convolution model
    fashion_model = Sequential()
    
    fashion_model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',
                     input_shape=(input_size,input_size,1), data_format="channels_last"))
    
    #fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='relu'))
    
    fashion_model.add(MaxPooling2D((2, 2)))
    fashion_model.add(Dropout(0.15))
    
    fashion_model.add(Conv2D(64, (3, 3), activation='relu'))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2)))
    fashion_model.add(Dropout(0.2))
    
    fashion_model.add(Conv2D(128, (3, 3), activation='relu'))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2)))
    fashion_model.add(Dropout(0.25))
    
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='relu'))
    fashion_model.add(Dropout(0.3))
    fashion_model.add(Dense(num_classes, activation='softmax'))
    
    print ("Create model successfully")
    if preCalculatedWeightPath:
        fashion_model.load_weights(preCalculatedWeightPath)
    
    fashion_model.summary()
    fashion_model.compile(loss=keras.losses.categorical_crossentropy, 
                          optimizer=keras.optimizers.Adam(),metrics=['accuracy'])  
    
    return fashion_model

