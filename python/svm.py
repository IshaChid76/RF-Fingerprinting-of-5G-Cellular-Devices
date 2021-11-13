from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def svm(x_train, y_train, kernel):
  scaler = StandardScaler()
  degree = 3
  # kernel = 'poly', degree=degree
  clf = None
  if kernel == 'rbf':
    clf = make_pipeline(scaler,SVC(gamma='auto',random_state=0))
  else:
    clf = make_pipeline(scaler,SVC(gamma='auto',kernel=kernel, degree=degree, random_state=0))
  clf.fit(x_train, y_train)
  return clf
