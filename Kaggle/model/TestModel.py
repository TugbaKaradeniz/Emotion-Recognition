# -*- coding: utf-8 -*-
"""
Created on Sun May 20 21:40:41 2018

@author: Tugba
"""
import numpy as np
from sklearn.metrics import classification_report
import sys
sys.path.append("../")

import model.CNN as objModel

final_emotions = [   'Angry',
                     'Fear',
                     'Happy',
                     'Sad',
                     'Surprise',
                     'Neutral']

def readData():
    test = np.load('../data/X_test.npy')
    label = np.load('../data/Y_test.npy')
    
    return test, label
             
def main():
    
    X, Y = readData()
#    print(Y.shape)
#    print(Y)
    Y = np.argmax(Y, axis=1)
#    print(Y.shape)
#    print(Y)
 
    model = objModel.buildModel('CNN_model_weights.h5')

    predicted_classes = model.predict(X)
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
    print(predicted_classes.shape, Y.shape)
    
    correct = np.where(predicted_classes==Y)[0]
    print("Found %d correct labels" % len(correct))
    incorrect = []  
    incorrect = np.where(predicted_classes!=Y)[0]
    print("Found %d incorrect labels" % len(incorrect))


    target_names = ["Class {}".format(final_emotions[i]) for i in range(len(final_emotions))]
    print(classification_report(Y, predicted_classes, target_names=target_names))     

if __name__ == '__main__':
    main()

