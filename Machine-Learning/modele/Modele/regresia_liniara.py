import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def regresia_liniara():
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)


    X_b = np.c_[np.ones((100, 1)), X]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)


    X_new = np.array([[0], [2]]) #iau un punct ce vreau sa il adaug, de ex (0,2)
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    y_predict = X_new_b.dot(theta_best)


    plt.plot(X_new, y_predict, "r-")
    plt.plot(X, y, "b.")
    plt.axis([0, 2, 0, 15])
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression")
    plt.show()
    
def regresia():
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print(lin_reg.intercept_, lin_reg.coef_)
    X_new = np.array([[0], [2]])
    predictii=lin_reg.predict(X_new)
    plt.scatter(X,y,marker='*')
    plt.plot(X_new,predictii,"r-")
    plt.axis([0, 2, 0, 15])
    plt.show()
    
    
def gradient_conjugat():
    np.random.seed(42)
    eta = 0.2
    n_epochs = 1000
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    X_1 = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
    n = 100

    w = np.random.randn(2,1)  # random initialization

    for epoch in range(n_epochs):
        for i in range(n):
            random_index = np.random.randint(n)
            xi = X_1[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = xi.T.dot(xi.dot(w) - yi)
            w = w - eta * gradients
    print(w)


def regresia_logistica():
    iris_dataset = datasets.load_iris()
    print(iris_dataset.keys())
    print(iris_dataset['target'])
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
    y_train_binary = (y_train == 2).astype(int) # 1 if iris virginica, else 0
    y_test_binary = (y_test == 2).astype(int) # 1 if iris virginica, else 0
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train_binary)
    print(log_reg.score(X_test, y_test_binary))
    log_reg.predict([X_test[0]])
    
    
    