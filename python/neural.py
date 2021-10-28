from keras.models import Sequential
from keras.layers import Dense

from fakedata import getData

model = Sequential()
model.add(Dense(60, input_dim=2, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


if __name__=='__main__':
  data = getData()
  X = []
  y = []
  for (i, waveform) in list(enumerate(data)):
    for wave in waveform:
      X.append(wave)
      y.append(i)
  model.fit(X, y, epochs=150, batch_size=10)
