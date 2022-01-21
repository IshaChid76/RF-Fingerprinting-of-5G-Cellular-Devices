import scipy.io
import pickle
import random
import numpy as np


def getData(points, method):
    x = []
    y = []
    for i in range(points):
        print(i)
        if method == 1:
            i = i + 1
            filename = '../matlab/tx' + str(i) + '.mat'
        elif method == 2:
            filename = "../matlab/5G_SDR_data_10Mpts/t" + \
                str(i)+"r"+str(i)+"_txf50_10M"
        mat = scipy.io.loadmat(filename)
        if method == 1:
            waveform = mat['x']
        elif method == 2:
            waveform = mat['rxdata']
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
    x = np.array(x)
    x = np.expand_dims(x, axis=3)
    temp = list(zip(x, y))
    random.seed(4)
    random.shuffle(temp)
    x, y = zip(*temp)
    return (x, y)


if __name__ == '__main__':
    getData(4, 2)
