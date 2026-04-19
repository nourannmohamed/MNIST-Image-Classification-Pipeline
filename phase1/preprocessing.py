import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from collections import Counter

def load_and_preprocess():
    # =========================
    # LOAD DATA
    # =========================
    (X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()

    X = np.concatenate([X_train_full, X_test_full], axis=0)
    y = np.concatenate([y_train_full, y_test_full], axis=0)

    print("Before preprocessing:", X.shape)

    # =========================
    # BINARY (0 vs ALL)
    # =========================
    y = (y == 0).astype(int)

    # =========================
    # NORMALIZE + FLATTEN
    # =========================
    X = X / 255.0
    X = X.reshape(X.shape[0], -1)

    # =========================
    # BALANCE DATASET (UNDERSAMPLING)
    # =========================
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]

    np.random.seed(42)
    idx_0_down = np.random.choice(idx_0, size=len(idx_1), replace=False)

    idx = np.concatenate([idx_0_down, idx_1])
    np.random.shuffle(idx)

    X, y = X[idx], y[idx]

    print("Balanced dataset:", Counter(y))

    # =========================
    # TRAIN TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # =========================
    # VARIANCE FILTER
    # =========================
    variances = np.var(X_train, axis=0)
    mask = variances > 0.01

    X_train = X_train[:, mask]
    X_test = X_test[:, mask]

    print("After preprocessing:", X_train.shape)

    return X_train, X_test, y_train, y_test