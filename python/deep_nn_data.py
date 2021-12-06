import scipy.io
import pickle
import random
import numpy as np


def getData(points):
    x = []
    y = []
    for i in range(points):
        print(i)
        i = i + 1
        filename = '../matlab/tx' + str(i) + '.mat'
        mat = scipy.io.loadmat(filename)
        waveform = mat['x']
        x_part = []

        for j, wave in enumerate(waveform):
            if(j % 160 == 0 and j != 0):
                x.append(x_part)
                y_part = [0] * points
                y_part[i-1] = 1
                y.append(y_part)
                x_part = []
            real = wave.real[0]
            imag = wave.imag[0]
            x_part.append([real, imag])
            # x_part.append(imag)
    # print(x)
    x = np.array(x)
    x = np.expand_dims(x, axis=3)
    temp = list(zip(x, y))
    random.seed(4)
    random.shuffle(temp)
    x, y = zip(*temp)
    return (x, y)
