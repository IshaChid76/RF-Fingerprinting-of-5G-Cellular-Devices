from sklearn.linear_model import LogisticRegression

def logistic(x_train, y_train):
  return LogisticRegression(max_iter=10000000, multi_class='ovr').fit(x_train, y_train)

# random forest, svm rbf kernel, ada boosting
# james kingsley computing server