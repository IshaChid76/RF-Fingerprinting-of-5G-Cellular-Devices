import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from threading import Thread
import pickle

from data_method1 import getData
#from deepnndata import getData, getData2
from ada import ada
from forest import forest
from logistic import logistic
from neural import neural
from svm import svm
from naiveBayes import naiveBayes
points = 15

conversion = {
    "ada": ada,
    "forest": forest,
    "logistic": logistic,
    "neural": lambda x, y: neural(x, y, points),
    "rbf svm": lambda x, y: svm(x, y, "rbf"),
    "poly svm": lambda x, y: svm(x, y, "poly"),
    "naive bernoulli": lambda x, y: naiveBayes(x, y, "bernoulli"),
    "naive gaussian": lambda x, y: naiveBayes(x, y, "gaussian"),
    "naive poly": lambda x, y: naiveBayes(x, y, "poly"),


}


times = 6
currentNetwork = [0] * 6


def runNetwork(i, x_train, x_test, y_train, y_test):
    clf = (conversion[net])(x_train, y_train)
    #print(clf.score(x_test, y_test))
    currentNetwork[i] = clf.score(x_test, y_test)
    ConfusionMatrixDisplay.from_predictions(y_test, clf.predict(x_test))
    plt.show()


if __name__ == '__main__':
    networks = ["ada", "forest", "logistic", "neural", "rbf svm",
                "poly svm", "naive bernoulli", "naive gaussian"]
    res = {}
    pointTests = [15]
    # use this to expand on prior data
    # when not using this form of data, res is used instead as a var
    '''with open('resdata.pkl', 'rb') as f:
        data = pickle.load(f)'''
    for points in pointTests:
        print("Points:", points)

        res[points] = {}
        x_trains = []
        x_tests = []
        y_trains = []
        y_tests = []
        for i in range(0, times):
            x, y = getData(points, 1, i)
            x_train, x_test = np.split(x, [int(len(x)*0.8)])
            y_train, y_test = np.split(y, [int(len(x)*0.8)])
            x_trains.append(x_train)
            x_tests.append(x_test)
            y_trains.append(y_train)
            y_tests.append(y_test)
        for net in networks:
            acc = 0
            print(net)
            res[points][net] = 0
            # data[points][net]=0
            threads = [None] * times
            for i in range(len(threads)):
                threads[i] = Thread(target=runNetwork, args=(
                    i, x_trains[i], x_tests[i], y_trains[i], y_tests[i]))
                threads[i].start()
            for i in range(len(threads)):
                threads[i].join()
            acc = sum(currentNetwork)/times
            print(acc)
            res[points][net] = acc
            #data[points][net] = acc
        # use this to store the results where you want
        # stores under
    '''with open('resdata.pkl', 'wb') as f:
        pickle.dump(data, f)'''
