import scipy.io
import pickle
import random
import numpy as np


def getData(points, method, seed):
    x = []
    y = []
    potentialPoints = []
    if method == 1:
        potentialPoints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    if method == 2:
        potentialPoints = [0, 1, 2, 3]
    random.seed(seed)
    random.shuffle(potentialPoints)
    for j in range(points):
        i = potentialPoints[j]
        print(i)
        if method == 1:
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
            if(j % 10000 == 0 and j != 0):
                x.append(x_part)
                y.append(i)
                x_part = []
            real = wave.real[0]
            imag = wave.imag[0]
            x_part.append(real)
            x_part.append(imag)
    x = np.array(x)
    temp = list(zip(x, y))
    random.seed(4)
    random.shuffle(temp)
    x, y = zip(*temp)
    return (x, y)


if __name__ == '__main__':
    data = getData(15)
    with open('test.pkl', 'wb') as f:
        pickle.dump(data, f)
