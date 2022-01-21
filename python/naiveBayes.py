from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

def naiveBayes(x_train, y_train, type="gaussian"):
    alg = None
    if type=="bernoulli":
        alg = BernoulliNB()
    elif type=="gaussian":
        alg = GaussianNB()
    else:
        alg = MultinomialNB()
    return alg.fit(x_train, y_train)
    #reference: https://scikit-learn.org/stable/modules/naive_bayes.html
    #gnb = GaussianNB()
    #y_pred = gnb.fit(X_train, y_train).predict(X_test)
