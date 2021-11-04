import scipy.io
import pickle
def getData(points):
  res = []
  for i in range(points):
    print(i)
    i = i + 1
    filename = '../matlab/tx' + str(i) + '.mat'
    mat = scipy.io.loadmat(filename)
    waveform = mat['x']
    x = []

    for j, wave in enumerate(waveform):
      if(j%10000==0 and j!= 0):
        res.append((x, i))
        x = []
      real = wave.real[0]
      imag = wave.imag[0]
      x.append([real, imag])
  return res

if __name__=='__main__':
  data = getData(15)
  with open('test.pkl','wb') as f:
    pickle.dump(data, f)