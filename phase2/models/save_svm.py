import os
import sys
import numpy as np

# add project root so phase2 imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from phase2.models.svm import MulticlassSVM

# load saved cnn features
X_train = np.load("phase2/feature_data/X_train.npy")
y_train = np.load("phase2/feature_data/y_train.npy")

# train svm model
print("Training SVM...")

# create svm with chosen parameters
model = MulticlassSVM(
    lr=0.001,
    lambda_param=0.001,
    n_iters=100
)

model.fit(X_train, y_train)

# create folder for saved models
os.makedirs("saved_models", exist_ok=True)

# save trained svm weights
np.save("saved_models/svm_weights.npy", model.W)

print("SVM weights saved successfully!")
