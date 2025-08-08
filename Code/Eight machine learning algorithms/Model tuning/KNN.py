import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams


# Read the dataset
file_path = "dataset"  # Replace with the path to your dataset
data = pd.read_csv(file_path)

# Extract features and target variable
X = data[[
    "absence of mental illness",
    "absence of physical illness",
    "cognitive health",
    "emotional health",
    "physical activity health",
    "sensory health",
    "sleep health",
    "subclinical health",
    "Years since publication"
]]  # Features
y = data['contribution index']

# Check if the target variable contains missing values
if y.isna().any():
    print("Warning: Missing values detected in the target variable. Dropping missing values!")
    y = y.dropna()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define parameter grid
param_grid = {
    'n_neighbors': [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

# Define the K-Nearest Neighbors regression model
knn_model = KNeighborsRegressor()

# Use GridSearchCV to find the best parameters, scoring function is R²
grid_search = GridSearchCV(
    estimator=knn_model,
    param_grid=param_grid,
    scoring='r2',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Run grid search
grid_search.fit(X_train, y_train)

# Output the best parameters
print("Best parameters:", grid_search.best_params_)
