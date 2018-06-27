# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 19:24:22 2018

@author: Tugba
"""
import cv2
import glob
import random
import numpy as np
import sys
sys.path.append("../")
# "contempt"
#Emotion list 
emotions = ["neutral", "anger",  "disgust",
            "fear", "happy", "sadness", "surprise"]

final_emotions = ["neutral", "anger", "fear", "happy", "sadness", "surprise"]

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("../data/CK+/dataset//%s//*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    
    return training, prediction

def make_sets(size):
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    
    print("---------------------------------------------------------")    
    for emotion in emotions:
        training, prediction = get_files(emotion)
        print("Number of sample for "+ emotion + " :")
        print("Training set has " + str(len(training)) + " samples.")
        print("Validation set has " + str(len(prediction)) + " samples.")
        print("---------------------------------------------------------")
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            out = cv2.resize(image, (size, size))
            training_data.append(out) #append image array to training data list     
            # "disgust" classified as "anger"
            if  emotions.index('disgust') == emotions.index(emotion):
                training_labels.append(final_emotions.index('anger'))     
            else:
                training_labels.append(final_emotions.index(emotion))
#            training_labels.append(final_emotions.index(emotion))

        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            out = cv2.resize(image, (size, size))
            prediction_data.append(out)          
            if  emotions.index('disgust') == emotions.index(emotion):
                prediction_labels.append(final_emotions.index('anger'))
            else:
                prediction_labels.append(final_emotions.index(emotion))
#            prediction_labels.append(final_emotions.index(emotion))
    
    X = np.array(training_data)        
    N = X.shape[0]
    X = X.reshape(N, size, size, 1)
    Y = np.array(training_labels)
#    print(X[0].shape)       
    
    X_test = np.array(prediction_data)       
    N_test = X_test.shape[0]
    X_test = X_test.reshape(N_test, size, size, 1)
    Y_test = np.array(prediction_labels)
    
    return X, Y, X_test, Y_test, (len(final_emotions))

#X, y, X_test, y_test, num_classes = make_sets(32)