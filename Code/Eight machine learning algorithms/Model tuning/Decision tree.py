import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler

# Read data
file_path = 'dataset'  # Replace with the path to your dataset
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
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define parameter grid
param_grid = {
    'max_depth': [4, 5, 8, 10, 12, 15, None],
    'min_samples_split': [3, 4, 5, 6],
    'min_samples_leaf': [6, 7, 8, 9, 10, 11, 13, 15, 20],
    'max_features': [None, 'sqrt', 'log2']
}

# Define the Decision Tree model
tree_model = DecisionTreeRegressor(random_state=42)

scorer = 'r2'

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(
    estimator=tree_model,
    param_grid=param_grid,
    scoring=scorer,
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Run grid search
grid_search.fit(X_train, y_train)

# Output the best parameters
print("Best parameters:", grid_search.best_params_)
