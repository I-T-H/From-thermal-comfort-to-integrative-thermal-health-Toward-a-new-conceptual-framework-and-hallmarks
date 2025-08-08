import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
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
    'alpha': [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 8.0, 9.0, 10.0]
}

# Ridge regression model
ridge_model = Ridge(random_state=42)

# Grid search
grid_search = GridSearchCV(
    estimator=ridge_model,
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
