import numpy as np
from collections import Counter



def euclidean_distance(a, b):
    # compute distance between two samples
    return np.sqrt(np.sum((a - b) ** 2))

class KNN:
    def __init__(self, k=5):
        self.k = k              #number of neighbors
        self.X_train = None     #training features
        self.y_train = None     #training labels

    def fit(self, X, y):
        # store training data
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # predict each sample
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # compute distance to all training samples
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]

        # choose most common label
        return Counter(k_labels).most_common(1)[0][0]
