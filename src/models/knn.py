import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class KNN:
    def __init__(self, k: int, ord: float):
        self.k_neighbor = k
        self.ord = ord
        self.X_train = None
        self.Y_train = None

    def fit(self, x, y):
        self.X_train = np.array(x)
        self.Y_train = np.array(y)

    def predict(self, x):
        x = np.array(x)
        pred = []
        for i, fits in enumerate(x):
            print(i)
            distances = np.linalg.norm(self.X_train - fits, ord=self.ord, axis=1)
            nearest_features = np.argsort(distances)[:self.k_neighbor]
            nearest_labels = self.Y_train[nearest_features]
            unique, counts = np.unique(nearest_labels, return_counts=True)
            pred.append(unique[np.argmax(counts)])
        return np.array(pred)
