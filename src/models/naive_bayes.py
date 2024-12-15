import numpy as np

class NaiveBayes:
    def __init__(self):
        self.class_priors = {}  # P(vj): Prior probabilities
        self.mean = {}  # Mean for numerical features per class
        self.var = {}  # Variance for numerical features per class
        self.classes = None

    def fit(self, X, y):
        """
        Train the Naive Bayes model on numerical data.
        X: np.ndarray - Feature matrix
        y: np.ndarray or pd.Series - Target labels
        """
        # Ensure X is a numpy array
        if isinstance(X, tuple):
            X = X[0]  # Extract the first element if X is a tuple
        X = np.array(X)  # Ensure X is a numpy array

        self.classes = np.unique(y)  # Unique classes in the target
        n_samples = len(y)

        for cls in self.classes:
            # Calculate priors P(vj)
            class_count = np.sum(y == cls)
            self.class_priors[cls] = class_count / n_samples

            # Subset data for the current class
            X_c = X[np.array(y == cls)]  # Convert boolean mask to numpy array for indexing

            # Calculate mean and variance for numerical features
            self.mean[cls] = X_c.mean(axis=0)
            self.var[cls] = X_c.var(axis=0)

    def _likelihood_num(self, class_idx, x):
        """
        Compute likelihood for numerical features using Gaussian distribution.
        class_idx: Class index
        x: Feature vector
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        likelihood = -((x - mean) ** 2) / (2 * var + 1e-6)  # Add small value to avoid division by zero
        likelihood = np.exp(likelihood) / np.sqrt(2 * np.pi * var + 1e-6)
        return likelihood.prod()

    def predict(self, X):
        """
        Predict the class for each sample in X.
        X: np.ndarray - Feature matrix
        """
        # Ensure X is a numpy array
        if isinstance(X, tuple):
            X = X[0]  # Extract the first element if X is a tuple
        X = np.array(X)  # Ensure X is a numpy array

        predictions = []
        for x in X:  # Iterate over each sample
            class_probs = {}
            for cls in self.classes:
                # Start with prior P(vj)
                class_prob = self.class_priors[cls]

                # Multiply with likelihoods for all features
                class_prob *= self._likelihood_num(cls, x)

                class_probs[cls] = class_prob

            # Assign the class with the highest probability
            predictions.append(max(class_probs, key=class_probs.get))
        return np.array(predictions)

    def accuracy(self, y_true, y_pred):
        """
        Calculate accuracy of the model.
        y_true: np.ndarray - True labels
        y_pred: np.ndarray - Predicted labels
        """
        return np.mean(y_true == y_pred)
