import numpy as np

class GaussianNB:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]

        self.mean = np.zeros((len(self.classes), n_features))
        self.var = np.zeros((len(self.classes), n_features))
        self.priors = np.zeros(len(self.classes))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]

            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-9  # avoid division by zero
            self.priors[idx] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])

            # Gaussian likelihood (log)
            likelihood = np.sum(self._log_pdf(idx, x))

            posterior = prior + likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _log_pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]

        numerator = - (x - mean) ** 2 / (2 * var)
        denominator = np.log(np.sqrt(2 * np.pi * var))

        return numerator - denominator