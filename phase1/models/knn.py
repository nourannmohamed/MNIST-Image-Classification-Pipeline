import numpy as np                      # numerical operations
from collections import Counter         # used for majority voting


def euclidean_distance(a, b):
    # calculate straight-line distance between two points
    return np.sqrt(np.sum((a - b) ** 2))


class KNN:

    def __init__(self, k=5):
        self.k = k                  # number of nearest neighbors
        self.X_train = None         # stores training features
        self.y_train = None         # stores training labels

    def fit(self, X, y):
        # save training data
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # predict label for every sample
        predictions = [self._predict(x) for x in X]

        # convert predictions to numpy array
        return np.array(predictions)

    def _predict(self, x):

        # compute distance from x to every training sample
        distances = [
            euclidean_distance(x, x_train)
            for x_train in self.X_train
        ]

        # sort distances and take nearest k samples
        k_indices = np.argsort(distances)[:self.k]

        # get labels of nearest neighbors
        k_labels = [self.y_train[i] for i in k_indices]

        # majority voting: most frequent class wins
        return Counter(k_labels).most_common(1)[0][0]