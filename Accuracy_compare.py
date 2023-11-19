import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

# Load the dataset
data = pd.read_csv('breast_cancer.csv')  # Replace with the actual path to our CSV file

# Prepare the data
X = data.drop(['id', 'diagnosis'], axis=1)  # Features
y = data['diagnosis']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM Model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_predictions)

# Logistic Regression Model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_predictions)

# Neural Network Model
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
nn_model.fit(X_train_scaled, y_train)
nn_predictions = nn_model.predict(X_test_scaled)
nn_accuracy = accuracy_score(y_test, nn_predictions)

# Compare Accuracies
print(f"SVM Accuracy: {svm_accuracy}")
print(f"Logistic Regression Accuracy: {lr_accuracy}")
print(f"Neural Network Accuracy: {nn_accuracy}")

# Compare Classification Reports
print("\nSVM Classification Report:")
print(classification_report(y_test, svm_predictions))

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions))

print("\nNeural Network Classification Report:")
print(classification_report(y_test, nn_predictions))
