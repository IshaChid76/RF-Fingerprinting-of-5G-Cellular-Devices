from typing import Reversible
import scipy.io

def getData():
  res = []
  for i in range(15):
    i = i + 1
    filename = '../matlab/tx' + str(i) + '.mat'
    mat = scipy.io.loadmat(filename)
    waveform = mat['x']
    xy = []
    for wave in waveform:
      real = wave.real[0]
      imag = wave.imag[0]
      xy.append([real, imag])
    #for j in range(len(waveform)):
    #print(type (waveform[0][0].real))
    #print(type (waveform[0][0].imag))
    res.append(xy)
  return res

if __name__=='__main__':
  getData()