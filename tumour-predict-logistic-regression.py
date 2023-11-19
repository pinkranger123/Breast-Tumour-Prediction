import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
url = 'your_dataset_file_path.csv'  # Replace with the actual path to your CSV file
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

# Logistic Regression from scratch
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, learning_rate, epochs):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    
    for epoch in range(epochs):
        z = np.dot(X, weights) + bias
        predictions = sigmoid(z)
        errors = predictions - y
        
        # Gradient Descent
        gradient = np.dot(X.T, errors) / m
        weights -= learning_rate * gradient
        bias -= learning_rate * np.sum(errors) / m
    
    return weights, bias

def predict_logistic_regression(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)

# Train the logistic regression model
learning_rate = 0.01
epochs = 1000
weights_logistic_regression, bias_logistic_regression = train_logistic_regression(X_train, y_train, learning_rate, epochs)

# SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Neural Network
nn_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
nn_model.fit(X_train, y_train)

# Evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

accuracy_logistic_regression = evaluate_model(weights_logistic_regression, X_test, y_test)
accuracy_svm = evaluate_model(svm_model, X_test, y_test)
accuracy_nn = evaluate_model(nn_model, X_test, y_test)

# Print accuracies
print(f"Logistic Regression Accuracy: {accuracy_logistic_regression:.4f}")
print(f"SVM Accuracy: {accuracy_svm:.4f}")
print(f"Neural Network Accuracy: {accuracy_nn:.4f}")
