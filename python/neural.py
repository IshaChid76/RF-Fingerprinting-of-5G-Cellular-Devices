'''from keras.clfs import Sequential
from keras.layers import Dense'''
from sklearn.neural_network import MLPClassifier

import numpy as np


def neural(x_train, y_train, points):
    '''clf = Sequential()
    clf.add(Dense(10000, input_dim=20000, activation='relu'))
    clf.add(Dense(points*100, activation='relu'))
    clf.add(Dense(points, activation='sigmoid'))

    clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])'''
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(points**2, points), max_iter=10000)
    clf.fit(x_train, y_train)
    return clf
