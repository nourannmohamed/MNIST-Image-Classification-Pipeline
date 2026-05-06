import os
import numpy as np

from phase2.models.svm import MulticlassSVM

# =====================================
# LOAD FEATURES
# =====================================
X_train = np.load("phase2/feature_data/X_train.npy")
y_train = np.load("phase2/feature_data/y_train.npy")

# =====================================
# TRAIN MODEL
# =====================================
print("Training SVM...")

model = MulticlassSVM(
    lr=0.001,
    lambda_param=0.01,
    n_iters=10
)

model.fit(X_train, y_train)

# =====================================
# SAVE WEIGHTS
# =====================================
os.makedirs("saved_models", exist_ok=True)

np.save("saved_models/svm_weights.npy", model.W)

print("SVM weights saved successfully!")