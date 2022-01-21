import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle


if __name__ == '__main__':
    pointTests = [2, 3, 4]
    res = {}
    x_pos_list = []
    keys = []
    for points in pointTests:
        with open('res'+str(points)+'data212.pkl', 'rb') as f:
            data = pickle.load(f)
            data = data[points]
        keys = []
        x = data.keys()
        y = []
        x_pos = [i for i, _ in enumerate(x)]
        x_pos_list.append(y)
        for key in data.keys():
            keys.append(key)
            y.append(data[key])
        plt.bar(x_pos, y)
        plt.xlabel("Classifier")
        plt.ylabel("Average accuracy")
        plt.title("The effect of different classifiers on fingerprinting " +
                  str(points) + " sources of 5G signals")
        plt.xticks(x_pos, x)
        plt.ylim([0, 1])

        plt.show()

    '''fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')'''

    '''table = ax.table(cellText=x_pos_list, rowLabels=pointTests,
                     colLabels=keys, loc="center")
    # table.set_fontsize(300)
    table.scale(4, 1)
    fig.tight_layout()'''
    # plt.show()
    #plt.rcParams["figure.figsize"] = (20, 10)
