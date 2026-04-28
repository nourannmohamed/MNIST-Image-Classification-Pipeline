import sys
import os
import numpy as np
# Add the project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from phase1.models.nb import GaussianNB

# load data
X_train = np.load("phase2/feature_data/X_train.npy")
X_test = np.load("phase2/feature_data/X_test.npy")
y_train = np.load("phase2/feature_data/y_train.npy")
y_test = np.load("phase2/feature_data/y_test.npy")

# train
model = GaussianNB()
model.fit(X_train, y_train)

# predict
preds = model.predict(X_test)

print("Done NB")