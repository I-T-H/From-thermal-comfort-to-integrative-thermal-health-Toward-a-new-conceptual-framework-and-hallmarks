import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.svm import SVR
import numpy as np
from sklearn.preprocessing import StandardScaler
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
    'C': [0.2, 0.3, 0.4, 0.5],
    'epsilon': [0.05, 0.1, 0.15, 0.2],
    'kernel': ['rbf', "poly"],
    'degree': [1, 2, 3, 4]
}

# Support Vector Regression model
svm_model = SVR()

# Scoring function (R²)
scorer = 'r2'

# Grid search for best parameters
grid_search = GridSearchCV(
    estimator=svm_model,
    param_grid=param_grid,
    scoring=scorer,
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Output best parameters
print("Best parameters:", grid_search.best_params_)
