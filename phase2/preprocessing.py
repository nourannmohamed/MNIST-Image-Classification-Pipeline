import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split


def load_and_preprocess(test_size=0.2, random_state=42):
    #load mnist train and test data
    (X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()

    #combine datasets to split them again like phase1
    X = np.concatenate([X_train_full, X_test_full], axis=0)
    y = np.concatenate([y_train_full, y_test_full], axis=0)

    print("Original dataset shape:", X.shape)

    #normalize pixels to [0, 1]
    X = X / 255.0

    #reshape images for cnn input
    X = X.reshape(-1, 28, 28, 1)

    #split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


#flatten images for non-cnn models
def get_flattened_data(X_train, X_test):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    return X_train_flat, X_test_flat


#run quick test for preprocessing
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess()

    print("Sample label:", y_train[0])
    print("Flattened shape example:",
    get_flattened_data(X_train, X_test)[0].shape)
