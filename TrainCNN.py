import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential # For constructing model
from tensorflow.keras.layers import Dense, Dropout, Flatten # Layer cores
from tensorflow.keras.layers import Conv2D, MaxPooling2D # CNN layers
from tensorflow.keras.utils import to_categorical # Extra utilities

import pickle
from sklearn.model_selection import train_test_split

import os

def loadData(fileName,size=0.2):

    with open(fileName, 'rb') as f:
        X, Y = pickle.load(f)
    
    X=X.reshape(-1,45,45,1)
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = size)
    return X_train, X_test, y_train, y_test


def createModel(input,output):

    model = Sequential()

    # Images are 48 by 48
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input)) #46 by 46 by 32
    model.add(Conv2D(64, (5,5), activation='relu')) #42 by 42 by 64
    model.add(MaxPooling2D((3,3),3)) #14 by 14 by 64
    model.add(Conv2D(80, (7,7), activation='relu')) #8 by 8 by 80
    model.add(MaxPooling2D((3,3),3)) #4 by 4 by 80
    model.add(Dropout(rate=0.15))
    model.add(Flatten()) #5120 by 1
    model.add(Dense(500, activation='relu')) #500 by 1
    model.add(Dropout(0.2))
    model.add(Dense(250, activation='relu')) #250 by 1
    model.add(Dropout(0.2))
    model.add(Dense(120, activation='relu')) #120 by 1
    model.add(Dropout(0.2))
    model.add(Dense(output, activation='softmax')) # 82 by 1 (only english, digits, and symbols)
    
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    return model
    
def trainModel(model, X_train, y_train, X_test, y_test, ep=50):

    checkpoint_path = "training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period = 2)
        
    model.save_weights(checkpoint_path.format(epoch=0))
    
    model.fit(X_train,
        y_train,
        epochs=ep,
        callbacks=[cp_callback],
        validation_data=(X_test,y_test),
        verbose=0)
        
    return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = loadData('X_Y_Data.pickle')
    model = createModel(X_train.shape[1:],np.max(y_train))
    model = trainModel(model, X_train, y_train, X_test, y_test)