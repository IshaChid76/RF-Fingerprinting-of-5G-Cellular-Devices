from sklearn.ensemble import RandomForestClassifier

import pickle
import numpy as np
import random
from fakedata import getData

if __name__=='__main__':
  points = 15
  x, y = getData(points)
  x_train, x_test = np.split(x, [int(len(x)*0.8)])
  y_train, y_test = np.split(y, [int(len(x)*0.8)])
  clf = RandomForestClassifier(random_state=0).fit(x_train, y_train)
  print(clf.predict(x_test))
  print(clf.score(x_test, y_test))