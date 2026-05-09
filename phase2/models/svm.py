import numpy as np


class MulticlassSVM:

    def __init__(self, lr=0.001, lambda_param=0.001, n_iters=100):

        self.lr = lr                    # controls update step size
        self.lambda_param = lambda_param   # regularization term to reduce overfitting
        self.n_iters = n_iters          # number of training epochs
        self.W = None                   # weight matrix for all classes

    def fit(self, X, y):

        # get dataset dimensions
        n_samples, n_features = X.shape

        # count number of unique classes
        n_classes = len(np.unique(y))

        # initialize weights with zeros
        self.W = np.zeros((n_classes, n_features))

        # training loop
        for _ in range(self.n_iters):

            # go through every training sample
            for i in range(n_samples):

                x_i = X[i]          # current input sample
                y_i = y[i]          # correct class label

                # compute score for every class
                scores = np.dot(self.W, x_i)

                # compute hinge loss margins
                margins = 1 + scores - scores[y_i]

                # ignore correct class margin
                margins[y_i] = 0

                # check which classes violate the margin condition
                violations = margins > 0

                # count number of violating classes
                count = np.sum(violations)

                # update wrong classes
                for r in range(n_classes):

                    # skip correct class
                    if r != y_i and violations[r]:

                        # push wrong class away
                        self.W[r] -= self.lr * (
                            x_i + self.lambda_param * self.W[r]
                        )

                # update correct class once
                self.W[y_i] += self.lr * (
                    count * x_i - self.lambda_param * self.W[y_i]
                )

    def predict(self, X):

        # compute class scores
        scores = np.dot(X, self.W.T)

        # choose class with highest score
        return np.argmax(scores, axis=1)