import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the dataset
url = 'breast_cancer.csv'  # Replace with the actual path to our CSV file
data = pd.read_csv(url)

# Preprocess the data
X = data.iloc[:, 2:].values
y = np.where(data['diagnosis'] == 'M', 1, 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM
class SVM:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for epoch in range(self.epochs):
            predictions = np.dot(X, self.weights) + self.bias
            errors = y * predictions
            for i, error in enumerate(errors):
                if error < 1:
                    self.weights += self.learning_rate * (y[i] * X[i] - 2 * (1 / self.epochs) * self.weights)
                    self.bias += self.learning_rate * y[i]

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)

# Train the SVM model
svm_model = SVM()
svm_model.fit(X_train, y_train)

# Predictions
y_pred_svm = svm_model.predict(X_test)

# Evaluate the model accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.4f}")

# Save the SVM model
np.save('svm_weights.npy', svm_model.weights)
np.save('svm_bias.npy', svm_model.bias)
