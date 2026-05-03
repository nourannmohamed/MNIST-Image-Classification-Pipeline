import sys
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from phase1.models.knn import KNN
from models.svm import MulticlassSVM
from models.perceptron import MulticlassPerceptron
from models.multinomial import MultinomialRegression


X = np.load("phase2/feature_data/X_train.npy")
y = np.load("phase2/feature_data/y_train.npy")


def cross_validate(model, X, y, folds=3):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        acc = accuracy_score(y_val, preds)
        scores.append(acc)

    return np.mean(scores)


def tune_knn():
    print("\n=== Tuning KNN ===")
    best_k = None
    best_score = 0

    for k in [1, 3, 5, 7, 9]:
        model = KNN(k=k)
        score = cross_validate(model, X, y)

        print(f"k={k} → {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"Best KNN: k={best_k}, score={best_score:.4f}")



def tune_svm():
    print("\n=== Tuning SVM ===")
    best_lambda = None
    best_score = 0

    for lam in [0.001, 0.01, 0.1]:
        model = MulticlassSVM(lambda_param=lam, lr=0.001, n_iters=100)
        score = cross_validate(model, X, y)

        print(f"lambda={lam} → {score:.4f}")

        if score > best_score:
            best_score = score
            best_lambda = lam

    print(f"Best SVM: lambda={best_lambda}, score={best_score:.4f}")


def tune_multinomial():
    print("\n=== Tuning Multinomial ===")
    best_lr = None
    best_score = 0

    for lr in [0.001, 0.01, 0.1]:
        model = MultinomialRegression(lr=lr, n_iters=300)
        score = cross_validate(model, X, y)

        print(f"lr={lr} → {score:.4f}")

        if score > best_score:
            best_score = score
            best_lr = lr

    print(f"Best Multinomial: lr={best_lr}, score={best_score:.4f}")


def tune_perceptron():
    print("\n=== Tuning Perceptron ===")
    best_lr = None
    best_score = 0

    for lr in [0.001, 0.01, 0.1]:
        model = MulticlassPerceptron(lr=lr, n_iters=200)
        score = cross_validate(model, X, y)

        print(f"lr={lr} → {score:.4f}")

        if score > best_score:
            best_score = score
            best_lr = lr

    print(f"Best Perceptron: lr={best_lr}, score={best_score:.4f}")


if __name__ == "__main__":
    tune_knn()
    tune_svm()
    tune_multinomial()
    tune_perceptron()