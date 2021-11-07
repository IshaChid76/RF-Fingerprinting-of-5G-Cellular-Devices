from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import pickle
import numpy as np
import random
from fakedata import getData

if __name__=='__main__':
  
  '''with open('test.pkl','rb') as f:
    data = pickle.load(f)'''
  points = 15
  data = getData(points)
  x = []
  y = []
  for part in data:
    xToAdd = []
    for xval in part[0]:
      for xvalpart in xval:
        xToAdd.append(xvalpart)
    x.append(xToAdd)
    y.append(part[1])
  scaler = StandardScaler()
  x = np.array(x)
  for i in range(10):
    temp = list(zip(x, y))
    random.seed(i)
    random.shuffle(temp)
    x, y = zip(*temp)
    x_train, x_test = np.split(x, [int(len(x)*0.8)])
    y_train, y_test = np.split(y, [int(len(x)*0.8)])
    clf = LogisticRegression(random_state=0, max_iter=10000000, multi_class='ovr').fit(x_train, y_train)
    print(clf.predict(x_test))
    print(clf.score(x_test, y_test))
