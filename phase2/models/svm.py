import numpy as np


class MulticlassSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=100):
        self.lr = lr        #learning rate
        self.lambda_param = lambda_param     #regularization strength  , Lambda controls overfitting
        self.n_iters = n_iters  #number of passess over dataset
        self.W = None           #weight matrix

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        self.W = np.zeros((n_classes, n_features))

        for _ in range(self.n_iters):
            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]

                scores = np.dot(self.W, x_i)

                # compute all margins
                margins = 1 + scores - scores[y_i]
                margins[y_i] = 0

                # find violating classes
                violations = margins > 0

                # count them
                count = np.sum(violations)

                # update wrong classes
                for r in range(n_classes):
                    if r != y_i and violations[r]:
                        self.W[r] -= self.lr * (x_i + self.lambda_param * self.W[r])

                # update correct class once
                self.W[y_i] += self.lr * (count * x_i - self.lambda_param * self.W[y_i])

    def predict(self, X):
        scores = np.dot(X, self.W.T)
        return np.argmax(scores, axis=1)