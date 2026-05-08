import numpy as np

class MultinomialRegression:
    def __init__(self, lr=0.1, n_iters=500):
        self.lr = lr            #learning rate
        self.n_iters = n_iters  #number of passess over dataset
        self.W = None           #weight matrix
        self.b = None           #bias for each class

    def softmax(self, z):
        #convert scores to probabilities
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  #stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def one_hot(self, y, num_classes):
        #make target vector for each label
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        #initialize weights and bias
        self.W = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)

        #convert labels to one hot form
        y_one_hot = self.one_hot(y, n_classes)

        for _ in range(self.n_iters):
            #compute scores for all classes
            scores = np.dot(X, self.W.T) + self.b

            #compute probability of each class
            probs = self.softmax(scores)

            #compare predicted probabilities with true labels
            error = probs - y_one_hot

            #compute gradients for weights and bias
            dW = (1 / n_samples) * np.dot(error.T, X)
            db = (1 / n_samples) * np.sum(error, axis=0)

            #update weights and bias
            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict(self, X):
        #compute class probabilities
        scores = np.dot(X, self.W.T) + self.b
        probs = self.softmax(scores)

        #choose class with highest probability
        return np.argmax(probs, axis=1)
