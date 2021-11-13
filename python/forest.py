from sklearn.ensemble import RandomForestClassifier

def forest(x_train, y_train):
  return RandomForestClassifier(random_state=0).fit(x_train, y_train)