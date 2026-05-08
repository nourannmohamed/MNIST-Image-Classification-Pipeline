
import numpy as np

class MulticlassPerceptron:
    def __init__(self, lr=0.001, n_iters=500):
        self.lr = lr            #learning rate
        self.n_iters = n_iters  #number of passess over dataset
        self.W = None           #weight matrix
        self.b = None           #bias for each class

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        #normalize features to [0, 1] so all pixels have same scale
        self.X_min = X.min(axis=0)
        self.X_max = X.max(axis=0)
        X = self._normalize(X)

        #random init instead of zeros
        rng = np.random.default_rng(42)
        self.W = rng.normal(0, 0.01, (n_classes, n_features))
        self.b = np.zeros(n_classes)

        for epoch in range(self.n_iters):
            #shuffle samples every epoch
            idx = rng.permutation(n_samples)

            #store updates for weights and bias
            dW = np.zeros_like(self.W)
            db = np.zeros_like(self.b)

            for i in idx:
                xi = X[i]
                yi = y[i]

                #compute scores for all classes
                scores = self.W @ xi + self.b
                score_true = scores[yi]

                #compare each class score with correct class score
                d = np.maximum(scores - score_true, 0)
                d[yi] = 0

                #if correct class is already best, no update
                Li = np.max(d)
                if Li == 0:
                    continue 

                #find the most wrong class
                r_star = np.argmax(d)

                #push correct class up and wrong class down
                dW[yi]     -= xi
                dW[r_star] += xi
                db[yi]     -= 1
                db[r_star] += 1

            #apply updates after each epoch
            self.W -= self.lr * dW
            self.b -= self.lr * db

            if (epoch + 1) % 50 == 0:
                #print training accuracy every 50 epochs
                preds = np.argmax(X @ self.W.T + self.b, axis=1)
                acc = np.mean(preds == y)
                print(f"Epoch {epoch+1}/{self.n_iters} — train acc: {acc:.4f}")

    def _normalize(self, X):
        #avoid division by zero for constant pixels
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1
        return (X - self.X_min) / denom

    def predict(self, X):
        #normalize test data same as training data
        X = self._normalize(X)
        scores = X @ self.W.T + self.b
        return np.argmax(scores, axis=1) 
