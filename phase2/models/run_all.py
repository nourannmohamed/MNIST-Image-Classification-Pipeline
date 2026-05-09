import sys
import os
import numpy as np


#add project root so phase1 imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from phase1.metrics import accuracy_score, confusion_matrix, classification_report
from phase1.models.knn import KNN
from phase1.models.nb import GaussianNB
from perceptron import MulticlassPerceptron
from multinomial import MultinomialRegression
from svm import MulticlassSVM

#load saved train and test data
X_train = np.load("phase2/feature_data/X_train.npy")
X_test  = np.load("phase2/feature_data/X_test.npy")
y_train = np.load("phase2/feature_data/y_train.npy")
y_test  = np.load("phase2/feature_data/y_test.npy")


#put all models in one dictionary
models = {
    "Naive Bayes": GaussianNB(),
    "Perceptron": MulticlassPerceptron(lr=0.001, n_iters=500),
    "Multinomial": MultinomialRegression(lr=0.1, n_iters=500),
    "SVM": MulticlassSVM(),
    "KNN": KNN(k=5)
}

#store accuracy for each model
results = {}

#evaluate each model
for name, model in models.items():
    
    print("\n" + "="*30)
    print(f"Model: {name}")
    print("="*30)

    #train model and predict test labels
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    #compute accuracy
    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(f"\nAccuracy: {acc:.4f}")

    # how true vs predicted classes
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    #show precision, recall, and f1-score
    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))


#print final accuracy comparison
print("\n=== Final Model Comparison ===")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")
