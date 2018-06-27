# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 01:11:49 2018

@author: Tugba
"""
import matplotlib.pyplot as plt
from keras.utils import np_utils as u
from load_data import make_sets
import CNN_2Conv as CNN

input_size = 32
batch_size = 32
epoch = 40

def main():
    
    #Lets start by loading the CK+ data
    X, y, X_test, y_test, num_classes = make_sets(input_size)
        
    X, X_test = X.astype('float32')/255.0, X_test.astype('float32')/255.0
    y, y_test = u.to_categorical(y), u.to_categorical(y_test)
    
    model = CNN.buildModel(None, input_size, num_classes)
    model.summary()
        
    train_dropout = model.fit(X, y, batch_size=batch_size, epochs=epoch,
                              verbose=1,validation_data=(X_test, y_test), shuffle=True)
        
    test_eval = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])
    print ("Training finished")
        
    model.save_weights("CNN(2Conv)_model_weights.h5")
        
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