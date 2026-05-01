import numpy as np
from multinomial import MultinomialRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load features
X_train = np.load("phase2/feature_data/X_train.npy")
X_test = np.load("phase2/feature_data/X_test.npy")
y_train = np.load("phase2/feature_data/y_train.npy")
y_test = np.load("phase2/feature_data/y_test.npy")

# train model
model = MultinomialRegression(lr=0.01, n_iters=500)
model.fit(X_train, y_train)

# predict
preds = model.predict(X_test)

# =========================
# METRICS
# =========================
print("\n==============================")
print("Model: Multinomial Regression")
print("==============================")

# Accuracy
acc = accuracy_score(y_test, preds)
print(f"\nAccuracy: {acc:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, preds))