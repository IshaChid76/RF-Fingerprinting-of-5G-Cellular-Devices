import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from fakedata import getData
from ada import ada
from forest import forest
from logistic import logistic
from neural import neural
from svm import svm

points = 8

conversion = {
  "ada": ada,
  "forest": forest,
  "logistic": logistic,
  "neural": lambda x, y :neural(x, y, points),
  "rbf svm": lambda x, y : svm(x, y, "rbf"),
  "poly svm": lambda x, y : svm(x, y, "poly"),
}
if __name__=='__main__':
  networks = ["ada", "forest","logistic","neural","rbf svm", "poly svm"]
  x, y = getData(points)
  x_train, x_test = np.split(x, [int(len(x)*0.8)])
  y_train, y_test = np.split(y, [int(len(x)*0.8)])
  for net in networks:
    print(net)
    clf = (conversion[net])(x_train, y_train)
    print(clf.predict(x_test))
    print(clf.score(x_test, y_test))
    ConfusionMatrixDisplay.from_predictions(y_test, clf.predict(x_test))
    plt.show()