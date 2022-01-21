import tensorflow as tf
from keras import Sequential
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, InputLayer, LeakyReLU, MaxPooling2D, Normalization
from keras.regularizers import l2
import numpy as np
import json
from data_method2 import getData
import pickle


def deepNeural(x_train, y_train, points, x_test, y_test, data):
    if not 'net' in data:
        poolSize = [2, 1]
        strideSize = [2, 1]
        numKnownRouters = points
        net = Sequential()
        net.add(InputLayer(
            input_shape=(160, 2, 1)))
        net.add(Normalization())
        net.add(Conv2D(50, kernel_size=(7, 1),
                input_shape=(160, 2, 1), padding='same'))

        net.add(BatchNormalization())
        net.add(LeakyReLU())
        net.add(MaxPooling2D(poolSize, strides=strideSize))
        net.add(Conv2D(50, kernel_size=(7, 2),
                input_shape=(160, 2, 1), padding='same'))
        net.add(BatchNormalization())
        net.add(LeakyReLU())
        net.add(MaxPooling2D(poolSize, strides=strideSize))

        '''net.add(Flatten())
        net.add(Dense(256, kernel_regularizer=l2(
            0.0001), bias_regularizer=l2(0.0001)))
        net.add(LeakyReLU())'''
        net.add(Dropout(rate=0.5))

        net.add(Flatten())
        net.add(Dense(80, kernel_regularizer=l2(
            0.0001), bias_regularizer=l2(0.0001)))
        net.add(LeakyReLU())
        net.add(Dropout(rate=0.5))

        net.add(Flatten())
        net.add(Dense(numKnownRouters, activation="softmax",
                kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001)))
        net.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001), metrics=['accuracy'])
        data['loss'] = []
        data['accuracy'] = []
        data['val_loss'] = []
        data['val_accuracy'] = []
    else:
        net = data['net']
    res = net.fit(x_train, y_train, epochs=3,
                  validation_data=(x_test, y_test))
    hist = res.history

    data['net'] = net
    data['loss'] = data['loss'] + hist['loss']
    data['accuracy'] = data['accuracy'] + hist['accuracy']
    data['val_loss'] = data['val_loss'] + ['val_loss']
    data['val_accuracy'] = data['val_accuracy'] + hist['val_accuracy']

    with open('deep_nn_historymeth2.pkl', 'wb') as f:
        pickle.dump(data, f)
    '''with open('deep_nn_history.txt', 'w') as file:
        file.write(json.dumps(res.history))'''
    # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])


if __name__ == '__main__':
    points = 4
    try:
        with open('deep_nn_historymeth2.pkl', 'rb') as f:
            data = pickle.load(f)
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']
        points = data['points']
        print(data['accuracy'])
        print(data['val_accuracy'])

    except:
        x, y = getData(points, 2)
        x_train, x_test = np.split(x, [int(len(x)*0.8)])
        y_train, y_test = np.split(y, [int(len(x)*0.8)])
        data = {}
        data['x_train'] = x_train
        data['x_test'] = x_test
        data['y_train'] = y_train
        data['y_test'] = y_test
        data['points'] = points

    # x is in shape 160 2 1 , 160 real imaginary pairs
    # y is in shape points, representing 0 1 0 0 if points = 4 and current y val = 2

    deepNeural(x_train, y_train, points, x_test, y_test, data)
