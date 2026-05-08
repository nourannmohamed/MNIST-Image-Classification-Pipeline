import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# add project folders to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from phase1.models.knn import KNN
from models.svm import MulticlassSVM
from models.perceptron import MulticlassPerceptron
from models.multinomial import MultinomialRegression
from phase1.models.nb import GaussianNB


# load saved cnn features
X_train = np.load("phase2/feature_data/X_train.npy")
X_test  = np.load("phase2/feature_data/X_test.npy")

# load train and test labels
y_train = np.load("phase2/feature_data/y_train.npy")
y_test  = np.load("phase2/feature_data/y_test.npy")


print("\n=== Learning Curves Analysis ===")


# different training sizes to test
train_sizes = [1000, 3000, 5000, 10000, len(X_train)]


# models to compare
models = {
    "Naive Bayes": GaussianNB(),
    "Perceptron": MulticlassPerceptron(lr=0.001, n_iters=200),
    "Multinomial": MultinomialRegression(lr=0.1, n_iters=300),
    "SVM": MulticlassSVM(lambda_param=0.001, lr=0.001, n_iters=100),
    "KNN": KNN(k=5)
}


# plot learning curve for each model
for model_name, model in models.items():

    print("\n" + "=" * 50)
    print(f"Model: {model_name}")
    print("=" * 50)

    train_accuracies = []
    test_accuracies = []

    for size in train_sizes:

        print(f"\nTraining with {size} samples")

        # take subset of training data
        X_subset = X_train[:size]
        y_subset = y_train[:size]

        # train model on current subset
        model.fit(X_subset, y_subset)

        # predict train and test labels
        train_preds = model.predict(X_subset)
        test_preds = model.predict(X_test)

        # compute accuracy
        train_acc = accuracy_score(y_subset, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Train Accuracy : {train_acc:.4f}")
        print(f"Test Accuracy  : {test_acc:.4f}")

        # compare train and test accuracy
        gap = train_acc - test_acc

        if gap > 0.05:
            print("Observation: Possible overfitting")

        elif train_acc < 0.80 and test_acc < 0.80:
            print("Observation: Possible underfitting")

        else:
            print("Observation: Good generalization")


    # create learning curve plot
    plt.figure(figsize=(8, 6))

    # plot training accuracy
    plt.plot(train_sizes, train_accuracies,
            marker='o', label='Training Accuracy')

    # plot testing accuracy
    plt.plot(train_sizes, test_accuracies,
            marker='o', label='Testing Accuracy')

    plt.title(f"Learning Curves - {model_name}")

    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.grid(True)

    plt.show()
