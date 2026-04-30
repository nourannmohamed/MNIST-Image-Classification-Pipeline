import sys
import os
import numpy as np

# =========================
# FIX PATH (IMPORTANT)
# =========================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# =========================
# IMPORT MODEL
# =========================
from phase2.models.svm import MulticlassSVM

# =========================
# METRICS
# =========================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =========================
# LOAD FEATURES
# =========================
X_train = np.load("phase2/feature_data/X_train.npy")
X_test = np.load("phase2/feature_data/X_test.npy")
y_train = np.load("phase2/feature_data/y_train.npy")
y_test = np.load("phase2/feature_data/y_test.npy")

print("Feature shape:", X_train.shape)

# =========================
# OPTIONAL: SPEED (DEBUG)
# =========================
# Uncomment if slow
# X_train = X_train[:10000]
# y_train = y_train[:10000]

# =========================
# TRAIN SVM
# =========================
print("\nTraining SVM...")

model = MulticlassSVM(lr=0.001, lambda_param=0.01, n_iters=10)
model.fit(X_train, y_train)

# =========================
# PREDICT
# =========================
print("\nPredicting...")

y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================
print("\n===== EVALUATION =====")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall   :", recall_score(y_test, y_pred, average='macro'))
print("F1 Score :", f1_score(y_test, y_pred, average='macro'))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# =========================
# LOAD ORIGINAL IMAGES
# =========================
(X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()

# Combine same as preprocessing
X_full = np.concatenate([X_train_full, X_test_full], axis=0)
y_full = np.concatenate([y_train_full, y_test_full], axis=0)

# Apply same split (IMPORTANT)
from sklearn.model_selection import train_test_split

_, X_test_images, _, y_test_images = train_test_split(
    X_full,
    y_full,
    test_size=0.2,
    stratify=y_full,
    random_state=42
)

# =========================
# VISUALIZE RESULTS
# =========================
n_samples = 10  # number of images to show

plt.figure(figsize=(12, 5))

for i in range(n_samples):
    plt.subplot(2, 5, i + 1)

    img = X_test_images[i]
    pred = y_pred[i]
    true = y_test_images[i]

    plt.imshow(img, cmap='gray')
    plt.title(f"P:{pred} / T:{true}",
              color='green' if pred == true else 'red')
    plt.axis('off')

plt.suptitle("SVM Predictions vs Ground Truth", fontsize=14)
plt.tight_layout()
plt.show()