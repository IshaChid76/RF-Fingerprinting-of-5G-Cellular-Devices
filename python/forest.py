from sklearn.ensemble import RandomForestClassifier

def forest(x_train, y_train):
  return RandomForestClassifier().fit(x_train, y_train)