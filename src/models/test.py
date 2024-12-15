from naive_bayes import NaiveBayes;
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
print("NILAI DARIIIIIII X")
print(X)

print("NILAII DARIIIIII Y")
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

nb = NaiveBayes()
nb.fit(X_train, y_train)


