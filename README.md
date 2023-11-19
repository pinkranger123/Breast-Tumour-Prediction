# Breast-Tumour-Prediction
Classifying breast tumour dataset into 2 classes i.e, benign or malignant using machine learning models like- Logistic Regression, SVM , Neural Network.

# Breast Tumor Classification Assignment

## Objective
1. Classify breast tumor dataset into 2 classes: benign or malignant using machine learning models (Logistic Regression, SVM, Neural Network).
2. Evaluate the model accuracy.

## Dataset
The dataset consists of fine needle aspiration biopsy reports, containing 569 biopsies (357 benign, 212 malignant). The 'diagnosis' column indicates the report labels (Malignant: M, Benign: B). The dataset includes 30 features computed for each cell nucleus.

## Code Files
1. `(https://tumour-predict-logistic-regression.py/)`: Code for Logistic Regression from scratch. 
2. `SVM.py`: Code for Support Vector Machine (SVM) from scratch.
3. `neural-network.py.py`: Code for Neural Network from scratch.
4. `DataExploration_Visualization.py`: Code for exploring and visualizing the dataset.

## Execution Instructions
1. Ensure Python is installed.
2. Install required libraries with `pip install pandas matplotlib seaborn scikit-learn tensorflow`.
3. Run each code file using `python filename.py`.


## Sample Dataset
Here is a snippet of the dataset:

```plaintext
id diagnosis radius_mean texture_mean perimeter_mean area_mean smoothness_mean ... fractal_dimension_worst
842302 M 17.99 10.38 122.8 1001 0.1184 ... 0.1189
842517 M 20.57 17.77 132.9 1326 0.08474 ... 0.08902
84300903 M 19.69 21.25 130 1203 0.1096 ... 0.08758
...
