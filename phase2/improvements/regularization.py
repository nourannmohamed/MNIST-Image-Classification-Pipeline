import sys
import os
import numpy as np

# add project folders to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from phase1.metrics import accuracy_score
from models.svm import MulticlassSVM

# load saved cnn features
X_train = np.load("phase2/feature_data/X_train.npy")
X_test  = np.load("phase2/feature_data/X_test.npy")

# load train and test labels
y_train = np.load("phase2/feature_data/y_train.npy")
y_test  = np.load("phase2/feature_data/y_test.npy")


print("\n=== Regularization Analysis (SVM) ===")


# different lambda values to test
lambda_values = [0.0001, 0.001, 0.01, 0.1, 1]

# store result for each lambda
results = []

for lam in lambda_values:

    print("\n" + "=" * 40)
    print(f"Testing lambda = {lam}")
    print("=" * 40)

    # create svm with current lambda
    model = MulticlassSVM(
        lambda_param=lam,
        lr=0.001,
        n_iters=100
    )

    # train svm
    model.fit(X_train, y_train)

    # predict train and test labels
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # compute accuracy
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    # compare train and test accuracy
    gap = train_acc - test_acc

    results.append((lam, train_acc, test_acc, gap))

    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Gap            : {gap:.4f}")

    # give simple diagnosis
    if gap > 0.05:
        print("Observation: Possible overfitting")
    elif train_acc < 0.80 and test_acc < 0.80:
        print("Observation: Possible underfitting")
    else:
        print("Observation: Good balance")


print("\n\n=== Final Regularization Comparison ===")

print(f"{'Lambda':<12}{'Train Acc':<15}{'Test Acc':<15}{'Gap'}")

# print final comparison table
for lam, train_acc, test_acc, gap in results:
    print(f"{lam:<12}{train_acc:<15.4f}{test_acc:<15.4f}{gap:.4f}")
