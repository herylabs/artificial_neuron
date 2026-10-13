"""

(X, y) -> Initialization(X) -> Model(X, W, b) -> Cost(A, y) -> Gradients(A, X, y) -> Update(W, b, dW, db)
                            -> Loooo.......................................ooooop ->
    step of the loop: LearningRate
    number of the loop: n_iter till the minimum of the Cost Function
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape(y.shape[0], 1)

print("shape of X: ", X.shape)
print("shape of y: ", y.shape)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="summer")
plt.show()


"""
    Initialization
"""
def Initialization(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

W, b = Initialization(X)
print("shape of W: ", W.shape)
print("shape of b: ", b.shape)


""" 
    Model 

    We use here the Sigmoid activation function 
    1 / (1 + exp(-z))
"""
def Model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

A = Model(X, W, b)
print(A)


"""
    Cost function
    Here we use Log Loss
"""
def LogLoss(A, y):
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

print(LogLoss(A, y))


"""" Gradients """
def Gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)

dW, db =  Gradients(A, X, y)
print("shape of dW: ", dW.shape)

""" Update """
def Update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b) 


""" Predict """
def Predict(X, W, b):
    A = Model(X, W, b)
    return A >= 0.5


""" Artificial Neuron """
def Neuron(X, y, learning_rate = 0.1, n_iter = 100):
    # Initialization
    W, b = Initialization(X)

    Loss = list()
    for i in range(n_iter):
        A = Model(X, W, b)
        Loss.append(LogLoss(A, y))
        dW, db = Gradients(A, X, y)
        W, b = Update(dW, db, W, b, learning_rate) 

    y_pred = Predict(X, W, b)
    print("accuracy score", accuracy_score(y_pred=y_pred, y_true=y))

    plt.plot(Loss)
    plt.show()

Neuron(X, y)


