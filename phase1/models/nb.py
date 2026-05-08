import numpy as np

class GaussianNB:
    def __init__(self):
        self.classes = None     #unique classes
        self.mean = None        #mean for each class
        self.var = None         #variance for each class
        self.priors = None      #class probabilities

    def fit(self, X, y):
        # get unique classes
        self.classes = np.unique(y)
        n_features = X.shape[1]

        # initialize statistics
        self.mean = np.zeros((len(self.classes), n_features))
        self.var = np.zeros((len(self.classes), n_features))
        self.priors = np.zeros(len(self.classes))

        for idx, c in enumerate(self.classes):
            # get samples for current class
            X_c = X[y == c]

            # compute mean, variance, and prior
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-9  # avoid division by zero
            self.priors[idx] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        # predict each sample
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # store posterior for each class
        posteriors = []

        for idx, c in enumerate(self.classes):
            # compute log prior
            prior = np.log(self.priors[idx])

            # compute gaussian likelihood in log form
            likelihood = np.sum(self._log_pdf(idx, x))

            # posterior = prior + likelihood
            posterior = prior + likelihood
            posteriors.append(posterior)

        # choose class with highest posterior
        return self.classes[np.argmax(posteriors)]

    def _log_pdf(self, class_idx, x):
        # get statistics for current class
        mean = self.mean[class_idx]
        var = self.var[class_idx]

        # gaussian probability formula in log form
        numerator = - (x - mean) ** 2 / (2 * var)
        denominator = np.log(np.sqrt(2 * np.pi * var))

        return numerator - denominator
