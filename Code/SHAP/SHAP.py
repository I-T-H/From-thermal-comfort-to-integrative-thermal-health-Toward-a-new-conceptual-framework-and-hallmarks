# Import necessary libraries
import random
import pandas as pd
import numpy as np
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams
from alepython import ale_plot
from PyALE import ale

# Load dataset
file_path = "dataset"
data = pd.read_csv(file_path)

# Features and target variable
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
    print("Warning: Missing values found in the target variable. Dropping missing values.")
    y = y.dropna()

print(f"Number of samples after removing outliers: {len(y)}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define XGBoost model
xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.7,
    colsample_bytree=0.4,
    random_state=42,
    alpha=0.1,
    reg_lambda=0.8
)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# SHAP analysis
print("\nPerforming SHAP analysis on the XGBoost model...")

# Create SHAP Explainer
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X)

# Remove "Years since publication" feature
X_shap_filtered = X.drop(columns=["Years since publication"])
shap_values_filtered = shap_values[:, :-1]

# Compute overall SHAP contribution (mean without absolute values)
shap_overall_contribution = shap_values_filtered.values.mean(axis=0)

# Create DataFrame for output
shap_contribution_df = pd.DataFrame({
    'Feature': X_shap_filtered.columns,
    'sum SHAP Value': shap_overall_contribution
}).sort_values(by='sum SHAP Value', ascending=False)

# Print SHAP overall contribution
print("\nSHAP Overall Contribution (sorted by mean):")
print(shap_contribution_df)

# Plot SHAP feature importance (dot plot)
shap.summary_plot(shap_values_filtered, X_shap_filtered, plot_type="dot", plot_size=(10, 8))

# Plot SHAP feature importance (bar plot)
shap.summary_plot(shap_values_filtered, X_shap_filtered, plot_type="bar", plot_size=(10, 8))
