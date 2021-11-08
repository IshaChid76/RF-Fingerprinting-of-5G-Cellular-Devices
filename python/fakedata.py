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
      if(j%10000==0 and j!= 0):
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

if __name__=='__main__':
  data = getData(15)
  with open('test.pkl','wb') as f:
    pickle.dump(data, f)