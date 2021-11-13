from sklearn.linear_model import LogisticRegression

def logistic(x_train, y_train):
  '''with open('test.pkl','rb') as f:
    data = pickle.load(f)'''
  return LogisticRegression(random_state=0, max_iter=10000000, multi_class='ovr').fit(x_train, y_train)

# random forest, svm rbf kernel, ada boosting
# james kingsley computing server