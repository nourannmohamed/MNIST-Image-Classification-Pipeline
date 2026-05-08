import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr            #learning rate
        self.n_iters = n_iters  #number of passess over dataset
        self.weights = None     #weights for features
        self.bias = None        #bias value

    def sigmoid(self, z):
        # convert value to probability
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # compute linear output
            linear_model = np.dot(X, self.weights) + self.bias

            # compute predicted probabilities
            y_pred = self.sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        # compute prediction probabilities
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)

        # convert probabilities to 0 or 1
        return (y_pred >= 0.5).astype(int)
