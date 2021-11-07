from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import numpy as np
import random
from fakedata import getData

if __name__=='__main__':
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
  temp = list(zip(x, y))
  random.seed(4)
  random.shuffle(temp)
  x, y = zip(*temp)
  x_train, x_test = np.split(x, [int(len(x)*0.8)])
  y_train, y_test = np.split(y, [int(len(x)*0.8)])
  clf = make_pipeline(StandardScaler(),SVC(gamma='auto',random_state=0)).fit(x_train, y_train)
  #clf = LogisticRegression(random_state=0, penalty='none', max_iter=10000000).fit(x_train, y_train)
  print(clf.predict(x_test))
  print(clf.score(x_test, y_test))
