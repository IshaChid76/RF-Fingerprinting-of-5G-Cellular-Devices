from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import pickle
import numpy as np
from fakedata import getData

if __name__=='__main__':
  
  '''with open('test.pkl','rb') as f:
    data = pickle.load(f)'''
  points = 3
  x, y = getData(points)
  scaler = StandardScaler()
  x_train, x_test = np.split(x, [int(len(x)*0.8)])
  y_train, y_test = np.split(y, [int(len(x)*0.8)])
  clf = LogisticRegression(random_state=0, max_iter=10000000, multi_class='ovr').fit(x_train, y_train)
  print(clf.predict(x_test))
  print(clf.score(x_test, y_test))

# random forest, svm rbf kernel, ada boosting
# james kingsley computing server