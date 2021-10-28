import scipy.io
filename = 'tx1.mat'
mat = scipy.io.loadmat(filename)
waveform = mat[x]
print(mat)