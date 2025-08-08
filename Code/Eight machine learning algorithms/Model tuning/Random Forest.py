import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from matplotlib import rcParams

# Load data
file_path = "dataset"  # Replace with your dataset file path
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
]]
y = data['contribution index']

# Check for missing values in the target variable
if y.isna().any():
    print("Warning: Missing values detected in the target variable. Missing values have been removed!")
    y = y.dropna()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Parameter grid
param_grid = {
    'n_estimators': [50, 100, 125, 150, 200],
    'max_depth': [10, 15, 20, 25, 30],
    'min_samples_split': [10, 20, 30, 40],
    'min_samples_leaf': [3, 5, 10, 15],
    'max_features': ['sqrt', 'log2']
}

# Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Grid search
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='r2',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit model
grid_search.fit(X_train, y_train)

# Output best parameters
print("Best parameters:", grid_search.best_params_)
