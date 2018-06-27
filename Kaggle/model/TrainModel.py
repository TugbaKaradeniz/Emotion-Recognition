# -*- coding: utf-8 -*-
"""
Created on Fri May 18 22:11:38 2018

@author: Tugba
"""

import matplotlib.pyplot as plt
import numpy as np
import CNN as CNN



final_emo_classes = ['Angry',
                     'Fear',
                     'Happy',
                     'Sad',
                     'Surprise',
                     'Neutral']

input_size = 48
batch_size = 32
epoch = 10

def main():
    
    #Lets start by loading the Kaggle data
    X = np.load('../data/X_train.npy')
    y = np.load('../data/Y_train.npy')
    x_val = np.load('../data/X_validation.npy')
    y_val = np.load('../data/Y_validation.npy')
    
    
    model = CNN.buildModel(None, input_size)
    model.summary()
        
    train_dropout = model.fit(X, y, batch_size=batch_size, epochs=epoch,
                              verbose=1,validation_data=(x_val, y_val), shuffle=True)
        
    test_eval = model.evaluate(x_val, y_val, verbose=1)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])
    print ("Training finished")
        
    model.save_weights("CNN_model_weights.h5")
        
    accuracy = train_dropout.history['acc']
    val_accuracy = train_dropout.history['val_acc']
        
    loss = train_dropout.history['loss']
    val_loss = train_dropout.history['val_loss']
    epochs = range(len(accuracy))
        
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
        
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()