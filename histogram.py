import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
url = 'breast_cancer.csv'  # Replace with the actual path to your CSV file
data = pd.read_csv(url)

# Select numerical columns for the histogram
numerical_columns = data.select_dtypes(include='number').columns

# Create histograms for each numerical feature
for column in numerical_columns:
    plt.figure(figsize=(8, 5))
    plt.hist(data[column], bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()
