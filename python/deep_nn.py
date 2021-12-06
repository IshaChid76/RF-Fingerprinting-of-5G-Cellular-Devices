import tensorflow as tf
from keras import Sequential
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, InputLayer, LeakyReLU, MaxPooling2D, Normalization
from keras.regularizers import l2
import numpy as np

from deep_nn_data import getData


def deepNeural(x_train, y_train, points, x_test, y_test):
    poolSize = [2, 1]
    strideSize = [2, 1]
    numKnownRouters = points
    print(x_train.shape)
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

    net.add(Flatten())
    net.add(Dense(256, kernel_regularizer=l2(
        0.0001), bias_regularizer=l2(0.0001)))
    net.add(LeakyReLU())
    net.add(Dropout(rate=0.0001))

    net.add(Flatten())
    net.add(Dense(80, kernel_regularizer=l2(
        0.0001), bias_regularizer=l2(0.0001)))
    net.add(LeakyReLU())
    net.add(Dropout(rate=0.0001))

    net.add(Flatten())
    net.add(Dense(numKnownRouters, activation="softmax",
            kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001)))
    net.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001), metrics=['accuracy'])
    net.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))


if __name__ == '__main__':
    points = 15
    x, y = getData(points)
    # x is in shape 160 2 1 , 160 real imaginary pairs
    # y is in shape points, representing 0 1 0 0 if points = 4 and current y val = 2
    x_train, x_test = np.split(x, [int(len(x)*0.8)])
    y_train, y_test = np.split(y, [int(len(x)*0.8)])
    deepNeural(x_train, y_train, points, x_test, y_test)
