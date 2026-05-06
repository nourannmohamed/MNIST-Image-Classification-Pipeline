import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.svm import MulticlassSVM

X_train = np.load("phase2/feature_data/X_train.npy")
X_test  = np.load("phase2/feature_data/X_test.npy")

y_train = np.load("phase2/feature_data/y_train.npy")
y_test  = np.load("phase2/feature_data/y_test.npy")


print("\n=== Regularization Analysis (SVM) ===")


lambda_values = [0.0001, 0.001, 0.01, 0.1, 1] #Different lambda values

results = []

for lam in lambda_values:

    print("\n" + "=" * 40)
    print(f"Testing lambda = {lam}")
    print("=" * 40)

    model = MulticlassSVM(   #Create model
        lambda_param=lam,
        lr=0.001,
        n_iters=100
    )

    model.fit(X_train, y_train)

    train_preds = model.predict(X_train) #Predictions
    test_preds = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)    #Accuracy
    test_acc = accuracy_score(y_test, test_preds)

    gap = train_acc - test_acc  #Overfitting gap

    results.append((lam, train_acc, test_acc, gap))

    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Gap            : {gap:.4f}")

    if gap > 0.05:      #Analysis
        print("Observation: Possible overfitting")
    elif train_acc < 0.80 and test_acc < 0.80:
        print("Observation: Possible underfitting")
    else:
        print("Observation: Good balance")


print("\n\n=== Final Regularization Comparison ===")

print(f"{'Lambda':<12}{'Train Acc':<15}{'Test Acc':<15}{'Gap'}")

for lam, train_acc, test_acc, gap in results:
    print(f"{lam:<12}{train_acc:<15.4f}{test_acc:<15.4f}{gap:.4f}")