import scipy.io
from deepNeural import neural

import numpy as np

def getData():
  filename = '../matlab/X_NNdataMatrix.mat'
  mat = scipy.io.loadmat(filename)
  return(mat['X'], mat['y'])

if __name__=='__main__':
  X, y = getData()
  print(X)
  print(y.shape)
  neural = neural(X, y, 15)