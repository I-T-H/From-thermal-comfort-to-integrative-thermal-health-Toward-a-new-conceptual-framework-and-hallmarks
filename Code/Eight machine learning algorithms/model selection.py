# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Load the dataset
file_path = "dataset"
data = pd.read_csv(file_path)

# Define features and target variable
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

# Check if the target variable has missing values
if y.isna().any():
    print("Warning: Missing values detected in the target variable. Missing values have been removed!")
    y = y.dropna()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns) # Standardized full dataset

X_train_scaled = scaler.fit_transform(X_train)  # Standardized training data
X_test_scaled = scaler.transform(X_test)       # Standardized testing data

# Define 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store models and results
models = {
    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.7,
        colsample_bytree=0.4,
        random_state=42,
        alpha=0.1,
        reg_lambda = 0.8
    ),
    "SVM": SVR(
        kernel='rbf',
        C=0.4,
        epsilon=0.1
    ),
    "KNN": KNeighborsRegressor(
        n_neighbors=28,
        weights='uniform',
        p=1
    ),
    "Ridge": Ridge(alpha=10.0, random_state=42),
    "Lasso": Lasso(alpha=0.01, max_iter=10000, random_state=42),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.01, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        max_features="sqrt",
        min_samples_split=20,
        min_samples_leaf=5,
        random_state=42
    ),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        max_features='sqrt',
        min_samples_split=3,
        min_samples_leaf=11,
        random_state=42
    )
}

results = {}  # Store the mean R² for each model

# Iterate over each model
for name, model in models.items():
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='r2')
    mean_r2 = np.mean(cv_scores)
    results[name] = mean_r2
    print(f"{name} 5-fold cross-validation mean R²: {mean_r2:.4f}")

# Find the model with the highest R²
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nThe model with the highest R² is: {best_model_name}, with a mean R² of: {results[best_model_name]:.4f}")

# Train the best model on the training set
best_model.fit(X_train_scaled, y_train)

# Predict on the testing set
y_pred_test = best_model.predict(X_test_scaled)
y_pred_train = best_model.predict(X_train_scaled)
r2_test = r2_score(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
print(f"\nBest model {best_model_name} training set R²: {r2_train:.4f}, testing set R²: {r2_test:.4f}")
