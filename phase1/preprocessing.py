import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from collections import Counter

def load_and_preprocess():
    # load mnist train and test data
    (X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()

    # combine train and test data
    X = np.concatenate([X_train_full, X_test_full], axis=0)
    y = np.concatenate([y_train_full, y_test_full], axis=0)

    print("Before preprocessing:", X.shape)

    # convert labels to binary 0 vs all
    y = (y == 0).astype(int)

    # normalize pixels to [0, 1]
    X = X / 255.0

    # flatten each image to one vector
    X = X.reshape(X.shape[0], -1)

    # get indices for both classes
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]

    # downsample larger class
    np.random.seed(42)
    idx_0_down = np.random.choice(idx_0, size=len(idx_1), replace=False)

    # combine balanced indices
    idx = np.concatenate([idx_0_down, idx_1])
    np.random.shuffle(idx)

    X, y = X[idx], y[idx]

    print("Balanced dataset:", Counter(y))

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # remove low-variance pixels
    variances = np.var(X_train, axis=0)
    mask = variances > 0.01

    # keep same useful pixels in train and test
    X_train = X_train[:, mask]
    X_test = X_test[:, mask]

    print("After preprocessing:", X_train.shape)

    return X_train, X_test, y_train, y_test
