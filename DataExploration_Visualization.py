import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('breast_cancer.csv')  # Replace with the actual path to our CSV file

# Select relevant columns for visualization
selected_features = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']

# Subset the data for selected features
selected_data = data[selected_features]

# Create pair plot
sns.pairplot(selected_data, hue='diagnosis', markers=['o', 's'], palette={'M': 'red', 'B': 'blue'})
plt.suptitle('Pair Plot of Selected Features', y=1.02)
plt.show()

# Create correlation heatmap
correlation_matrix = selected_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Selected Features')
plt.show()
