# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:01:53 2018

@author: Tugba
"""

from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np
import random

# This file separate training and validation data. While generating data, we classified Disgust as Angry.
# So resulting data will contains 6-class balanced dataset that contains Angry, Fear, Happy, Sad, Surprise and Neutral
# fer2013 dataset:
# It comprises a total of 35887 pre-cropped, 48-by-48-pixel grayscale images of faces each
# labeled with one of the 7 emotion classes: anger, disgust, fear, happiness, sadness, surprise, and neutral.
# Training       28709
# PublicTest      3589
# PrivateTest     3589


# emotion labels from FER2013:
original_emo_classes = {'Angry': 0,
                        'Disgust': 1,
                        'Fear': 2,
                        'Happy': 3,
                        'Sad': 4,
                        'Surprise': 5,
                        'Neutral': 6}
final_emo_classes = ['Angry',
                     'Fear',
                     'Happy',
                     'Sad',
                     'Surprise',
                     'Neutral']

filepath='../data/fer2013.csv'
file = pd.read_csv(filepath)

# Reconstruct original image to size 48X48. Returns numpy array of image pixels
def fnReconstruct(original_pixels, size=(48, 48)):
    arrPixels = []
    for pixel in original_pixels.split():
        arrPixels.append(int(pixel))
    arrPixels = np.asarray(arrPixels)
    return arrPixels.reshape(size)

#This function merge disgust emotion label to anger label and returns count of each emotion class
def fnGetEmotionCount(y_train, emoClasses, verbose=True):
    emo_classcount = {}
    #fer2013 dataset contains only 113 samples of "disgust" class compared to many other classes.
    #Therefore we merge disgust into anger to prevent this imbalance.
#    print ('Disgust classified as Angry')
    y_train.loc[y_train == 1] = 0
    emoClasses.remove('Disgust')
    for newNum, className in enumerate(emoClasses):
        y_train.loc[(y_train == original_emo_classes[className])] = newNum
        class_count = sum(y_train == (newNum))
        if verbose:
            print ('{}: {} with {} samples'.format(newNum, className, class_count))
        emo_classcount[className] = (newNum, class_count)
    return y_train.values, emo_classcount

#loads data from fer2013.csv
def fnLoadData(Sample_split_fraction, usage, boolCategorize=True, verbose=True,
               default_classes=final_emo_classes):
    # read .csv file using pandas library

    df = file[file.Usage == usage]
    arrFrames = []
    default_classes.append('Disgust')
    for _class in default_classes:
        class_df = df[df['emotion'] == original_emo_classes[_class]]
        arrFrames.append(class_df)
    data = pd.concat(arrFrames, axis=0)
    rows = random.sample(list(data.index), int(len(data) * Sample_split_fraction))
    data = data.loc[rows]
    print ('{} set for {}: {}'.format(usage, default_classes, data.shape))
    data['pixels'] = data.pixels.apply(lambda x: fnReconstruct(x))
    x = np.array([mat for mat in data.pixels])
    X = x.reshape(-1, 1, x.shape[1], x.shape[2])
    Y, new_dict = fnGetEmotionCount(data.emotion, default_classes, verbose)
#    print (new_dict)
    if boolCategorize:
        Y = to_categorical(Y)
        
    print(X.shape)
    print(Y.shape)
    print("******************************************************")
    
    return X, Y, new_dict

# Save X_train (images) and Y_train (labels) to local folder for training
def fnSaveData(X, Y, fname='', folder='../data/'):
    np.save(folder + 'X' + fname, X)
    np.save(folder + 'Y' + fname, Y)

if __name__ == '__main__':
    # makes the numpy arrays ready to use:
    print ('Making moves...')

    X_train, Y_train, emo_dict = fnLoadData(Sample_split_fraction=0.02,
                                            usage='Training',
                                            verbose=True)
    
    X_val, Y_val, emo_dict = fnLoadData(Sample_split_fraction=0.03,
                                            usage='PrivateTest',
                                            verbose=True)
    
    X_test, Y_test, emo_dict = fnLoadData(Sample_split_fraction=0.03,
                                            usage='PublicTest',
                                            verbose=True)
    print ('Saving...')
    
    fnSaveData(X_train, Y_train, fname='_train')
    print(X_train.shape)
    print(Y_train.shape)
   
    fnSaveData(X_val, Y_val, fname='_validation')
    print(X_val.shape)
    print(Y_val.shape)

    fnSaveData(X_test, Y_test, fname='_test')

    print ('Done!')

