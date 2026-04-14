import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter

from main import X_train, X_test, y_train, y_test, X, y

# =========================
# PERCEPTRON
# =========================
def unit_step_func(x):
    return np.where(x > 0, 1, 0)

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            indices = np.random.permutation(n_samples)
            for idx in indices:
                x_i = X[idx]
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = unit_step_func(linear_output)

                update = self.lr * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return unit_step_func(linear_output)


# =========================
# TRAIN
# =========================
print("\nTraining Perceptron...")
model = Perceptron(learning_rate=0.01, n_iters=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nPrediction distribution:", Counter(y_pred))

# =========================
# EVALUATION
# =========================
print("\n===== EVALUATION =====")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# =========================
# PCA (2D FOR VISUALIZATION)
# =========================
mean = np.mean(X_train, axis=0)
X_centered = X_train - mean

cov = np.cov(X_centered, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)

sort_idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, sort_idx]

W_2D = eigvecs[:, :2]
X_2D = X_centered @ W_2D

# =========================
# PLOT DATA — PCA SCATTER
# =========================
plt.figure(figsize=(8, 6))
plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y_train, alpha=0.5)
plt.title("MNIST (0 vs ALL) - PCA 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.show()

# =========================
# DECISION BOUNDARY
# =========================
w_2D = model.weights @ W_2D
b = model.bias

x_min, x_max = X_2D[:, 0].min() - 1, X_2D[:, 0].max() + 1
y_min, y_max = X_2D[:, 1].min() - 1, X_2D[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]
Z = (grid @ w_2D + b > 0).astype(int).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y_train,
            cmap='RdYlBu', edgecolor='k', linewidths=0.3, s=15)
plt.title("Perceptron Decision Boundary")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Class (0 = Not Zero, 1 = Zero)")
plt.tight_layout()
plt.show()

# =========================
# CONFUSION MATRIX VISUAL
# =========================
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["Not 0", "0"])
plt.yticks([0, 1], ["Not 0", "0"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=13)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# =========================
# SHOW SAMPLE IMAGES
# =========================
zeros_idx    = np.where(y == 1)[0][:5]
nonzeros_idx = np.where(y == 0)[0][:5]
sample_idx   = np.concatenate([zeros_idx, nonzeros_idx])

plt.figure(figsize=(10, 4))
for i, idx in enumerate(sample_idx):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[idx].reshape(28, 28), cmap='gray')
    plt.title("Digit 0" if y[idx] == 1 else "Not 0")
    plt.axis('off')

plt.suptitle("Sample Images: Actual Zeros vs Non-Zeros", fontsize=13)
plt.tight_layout()
plt.show()