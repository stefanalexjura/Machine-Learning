from sklearn import datasets
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

iris_dataset=datasets.load_iris()

X = iris_dataset["data"][:, (2, 3)]  # petal length, petal width
y = iris_dataset["target"]

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

svm_clf = SVC(kernel="linear", C=1e9)
svm_clf.fit(X, y)

x0, x1 = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
)
X_new = np.c_[x0.ravel(), x1.ravel()]
y_predict = svm_clf.predict(X_new).reshape(x0.shape)
plt.figure(figsize=(10, 6))
plt.contourf(x0, x1, y_predict, alpha=0.2)


plt.scatter(svm_clf.support_vectors_[:, 0], svm_clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')


plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("SVM Decision Boundary with Support Vectors")
plt.show()
