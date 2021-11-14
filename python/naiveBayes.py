from sklearn.naive_bayes import GaussianNB

def naiveBayes(x_train, y_train):
    return GaussianNB().fit(x_train, y_train)
    #reference: https://scikit-learn.org/stable/modules/naive_bayes.html
    #gnb = GaussianNB()
    #y_pred = gnb.fit(X_train, y_train).predict(X_test)
