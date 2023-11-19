import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = 'breast_cancer.csv'  # Replace with the actual path to our CSV file
data = pd.read_csv(url)

# Display basic information about the dataset
print(data.info())

# Display the first few rows of the dataset
print(data.head())

# Summary statistics of numerical columns
print(data.describe())

# Visualize the distribution of the target variable 'diagnosis'
sns.countplot(x='diagnosis', data=data)
plt.title('Distribution of Diagnosis (Malignant and Benign)')
plt.show()

# Visualize the correlation matrix of numerical features
plt.figure(figsize=(15, 12))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
