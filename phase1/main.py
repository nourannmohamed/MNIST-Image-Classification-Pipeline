from preprocessing import load_and_preprocess
from models.nb import GaussianNB
from models.perceptron import Perceptron
from models.knn import KNN
from models.logistic import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
X_train, X_test, y_train, y_test = load_and_preprocess()

models = {
    "Naive Bayes": GaussianNB(),
    "Perceptron": Perceptron(),
    "KNN": KNN(k=5),
    "Logistic Regression": LogisticRegression()
}

results = {}

for name, model in models.items():
    print("\n" + "="*30)
    print(f"Model: {name}")
    print("="*30)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"\nAccuracy: {acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

print("\n=== Final Model Comparison ===")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")