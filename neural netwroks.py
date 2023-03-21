import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = [0, 0, 1, 1]

clf = MLPClassifier(random_state=1, max_iter=1000)
clf.fit(X, y)
print(clf.score(X, y))
print(clf.predict([[2, 2]]))
