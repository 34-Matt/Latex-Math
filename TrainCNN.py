import tensorflow as tf
from tensorflowimport keras
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
        
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = size)
    return X_train, X_test, y_train, y_test


def createModel(size):

    model = Sequential()

    # Images are 48 by 48
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=size)) #46 by 46
    model.add(Conv2D(32, (3,3), activation='relu')) #44 by 44
    model.add(Dropout(rate=0.15))
    model.add(Flatten()) #1964 by 1
    model.add(Dense(500, activation='relu')) #500 by 1
    model.add(Dropout(0.2))
    model.add(Dense(250, activation='relu')) #250 by 1
    model.add(Dropout(0.2))
    model.add(Dense(120, activation='relu')) #120 by 1
    model.add(Dropout(0.2))
    model.add(Dense(82, activation='softmax')) # 82 by 1 (only english, digits, and symbols)
    
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

if __name__ = "__main__":
    X_train, X_test, y_train, y_test = loadData('X_Y_Data.pickle')
    model = createModel(X_train.shape[1:])
    model = trainModel(model, X_train, y_train, X_test, y_test)