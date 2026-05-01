import numpy as np

class MultinomialRegression:
    def __init__(self, lr=0.01, n_iters=500):
        self.lr = lr
        self.n_iters = n_iters
        self.W = None   # weights (classes x features)
        self.b = None   # bias (classes)

    # -----------------------------
    # SOFTMAX
    # -----------------------------
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # -----------------------------
    # ONE HOT ENCODING
    # -----------------------------
    def one_hot(self, y, num_classes):
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    # -----------------------------
    # TRAIN
    # -----------------------------
    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # initialize weights
        self.W = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)

        y_one_hot = self.one_hot(y, n_classes)

        for _ in range(self.n_iters):

            # scores (v_r = W_r^T X)
            scores = np.dot(X, self.W.T) + self.b

            # softmax probabilities
            probs = self.softmax(scores)

            # gradient (matches tutorial)
            error = probs - y_one_hot

            dW = (1 / n_samples) * np.dot(error.T, X)
            db = (1 / n_samples) * np.sum(error, axis=0)

            # update
            self.W -= self.lr * dW
            self.b -= self.lr * db

    # -----------------------------
    # PREDICT
    # -----------------------------
    def predict(self, X):
        scores = np.dot(X, self.W.T) + self.b
        probs = self.softmax(scores)
        return np.argmax(probs, axis=1)