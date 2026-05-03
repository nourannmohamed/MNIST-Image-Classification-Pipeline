import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from phase1.models.knn import KNN
from phase1.models.nb import GaussianNB
from perceptron import MulticlassPerceptron
from multinomial import MultinomialRegression
from svm import MulticlassSVM

X_train = np.load("phase2/feature_data/X_train.npy")
X_test  = np.load("phase2/feature_data/X_test.npy")
y_train = np.load("phase2/feature_data/y_train.npy")
y_test  = np.load("phase2/feature_data/y_test.npy")


models = {
    "KNN": KNN(k=5),
    "Naive Bayes": GaussianNB(),
    "Perceptron": MulticlassPerceptron(lr=0.01, n_iters=500),
    "Multinomial": MultinomialRegression(lr=0.01, n_iters=500),
    "SVM": MulticlassSVM()
}

results = {}


for name, model in models.items():
    print("\n" + "="*30)
    print(f"Model: {name}")
    print("="*30)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(f"\nAccuracy: {acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))


print("\n=== Final Model Comparison ===")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")