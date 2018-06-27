# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 14:51:34 2018

@author: Tugba
"""
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import random
import sys
sys.path.append("../")

import model.CNN_2Conv as objModel

final_emotions = ["neutral", "anger", "fear", "happy", "sadness", "surprise"]

FACE_SHAPE = (32, 32)
faceDet = cv2.CascadeClassifier ("../webcam/haarcascade_frontalface_default.xml")
test = []
label = []

model = objModel.buildModel('CNN(2Conv)_model_weights.h5')
incorrect = []

def readData():
#    total = 0
    print("---------------------------------------------------------") 
    for emotion in final_emotions:
        files = glob.glob("../sample_image_directory//%s//*" %emotion)
#        total = total + len(files)
        print("Number of sample for "+ emotion + " :")
        print("Test set has " + str(len(files)) + " samples.")
        print("---------------------------------------------------------")
        
        for f in files:
            img = cv2.imread(f, 0)
            face  = faceDet.detectMultiScale(img, scaleFactor=1.1, minNeighbors=10, 
                                             minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            if len(face) == 1:
                facefeatures = face
            else:
                facefeatures = ""
            
            for (x, y, w, h) in facefeatures: 
                img = img[y:y+h, x:x+w]
            img = cv2.resize(img, FACE_SHAPE)
            test.append(img)
            label.append(final_emotions.index(emotion))
             
def main():
    readData()
    X = np.array(test)
    n = X.shape[0]
    X = X.reshape(n, 32, 32, 1)
    Y = np.array(label)
    
    predicted_classes = model.predict(X)
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
    print(predicted_classes.shape, Y.shape)
    
    correct = np.where(predicted_classes==Y)[0]
    print("Found %d correct labels" % len(correct))
    for i, correct in enumerate(correct[:9]):
        plt.subplot(3,3,i+1)
        plt.imshow(X[correct].reshape(32,32), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[correct], Y[correct]))
        plt.tight_layout()
        
    incorrect = np.where(predicted_classes!=Y)[0]
    print("Found %d incorrect labels" % len(incorrect))
    for i, incorrect in enumerate(random.sample(list(incorrect),9)):
        plt.subplot(3,3,i+1)
        plt.imshow(X[incorrect].reshape(32,32), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], Y[incorrect]))
        plt.tight_layout()

    target_names = ["Class {}".format(final_emotions[i]) for i in range(len(final_emotions))]
    print(classification_report(Y, predicted_classes, target_names=target_names))     

if __name__ == '__main__':
    main()

