from sklearn.ensemble import AdaBoostClassifier

def ada(x_train, y_train):
  return AdaBoostClassifier(random_state=0, n_estimators=100).fit(x_train, y_train)