import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

from phase1.main import X_train, X_test, y_train, y_test

#Distance function
def euclidean_distance(a,b):
    return np.sqrt(np.sum((a-b) ** 2) )

#KNN class
class KNN:
    def __init__(self,k=5):
        self.k=k

    def fit(self,X,y):
        self.X_train=X
        self.y_train=y

    #Step-by-step:Loop over each test point, Predict its label, Store results, Convert list to numpy array
    def predict (self,x):
        predictions=[self._predict(x) for x in x]
        return np.array(predictions)

    def _predict(self, x):
        # Compute distances between test point x and all training points
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get k nearest
        k_indices = np.argsort(distances)[:self.k]          #Sort distances (smallest first), Take first k indices
        k_labels = [self.y_train[i] for i in k_indices]     #Get their labels

        # Majority vote
        return Counter(k_labels).most_common(1)[0][0]


# TRAIN + TEST
print("\nTraining KNN...")

model = KNN(k=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# EVALUATION
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("\n===== EVALUATION =====")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)