import numpy as np

def unit_step_func(x):
    # convert output to 0 or 1
    return np.where(x > 0, 1, 0)

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=100):
        self.lr = learning_rate  #learning rate
        self.n_iters = n_iters   #number of passess over dataset
        self.weights = None      #weights for features
        self.bias = None         #bias value

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # shuffle samples every epoch
            indices = np.random.permutation(n_samples)

            for idx in indices:
                x_i = X[idx]

                # compute linear output
                linear_output = np.dot(x_i, self.weights) + self.bias

                # predict class
                y_pred = unit_step_func(linear_output)

                # compute update amount
                update = self.lr * (y[idx] - y_pred)

                # update weights and bias
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        # compute output for all samples
        linear_output = np.dot(X, self.weights) + self.bias

        # convert output to 0 or 1
        return unit_step_func(linear_output)
