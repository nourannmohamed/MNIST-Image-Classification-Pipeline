from preprocessing import load_and_preprocess
from models.nb import GaussianNB
from models.perceptron import Perceptron
from models.knn import KNN
from models.logistic import LogisticRegression
from metrics import accuracy_score, confusion_matrix, classification_report

# load preprocessed binary mnist data
X_train, X_test, y_train, y_test = load_and_preprocess()

# put all models in one dictionary
models = {
    "Naive Bayes": GaussianNB(),
    "Perceptron": Perceptron(),
    "KNN": KNN(k=5),
    "Logistic Regression": LogisticRegression()
}

# store accuracy for each model
results = {}

# evaluate each model
for name, model in models.items():
    # print model name
    print("\n" + "="*30)
    print(f"Model: {name}")
    print("="*30)

    # train model and predict test labels
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # compute accuracy
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"\nAccuracy: {acc:.4f}")

    # show true vs predicted classes
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # show precision, recall, and f1-score
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# print final accuracy comparison
print("\n=== Final Model Comparison ===")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")
