
import numpy as np
import os
from perceptron import MulticlassPerceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "feature_data")

# load features
X_train = np.load(os.path.join(DATA, "X_train.npy"))
X_test  = np.load(os.path.join(DATA, "X_test.npy"))
y_train = np.load(os.path.join(DATA, "y_train.npy"))
y_test  = np.load(os.path.join(DATA, "y_test.npy"))

# train model
model = MulticlassPerceptron(lr=0.01, n_iters=500)
model.fit(X_train, y_train)

# predict
preds = model.predict(X_test)

# =========================
# METRICS
# =========================
print("\n==============================")
print("Model: Multiclass Perceptron")
print("==============================")

acc = accuracy_score(y_test, preds)
print(f"\nAccuracy: {acc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

print("\nClassification Report:")
print(classification_report(y_test, preds, zero_division=0)) 