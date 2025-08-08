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
from alepython import ale_plot  # Import alepython package
import PyALE
from PyALE import ale


# Read the dataset
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

# Define the XGBoost model
xgb_model = XGBRegressor(
    n_estimators=125,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
    alpha=0.5,
    reg_lambda=0.4
    )

# Train the model on the training set
xgb_model.fit(X_train, y_train)

# Predict on the testing set
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# # # ---------------- Perform ALE Analysis using alepython ----------------
print("\nPerforming ALE analysis...")

# Iterate over each feature and plot ALE (excluding "Years since publication")
for feature in X.columns:
    if feature != "Years since publication":  # Exclude "Years since publication"
        print(f"Plotting ALE for feature: {feature}")

        # Set figure size
        mpl.rc("figure", figsize=(6, 6))

        # Plot ALE using alepython with the specified format
        ale_plot(
            xgb_model,  # Pass the machine learning model (e.g., trained regression or classification model)
            X_train,  # Feature dataset for generating ALE plots
            features=feature,
            bins=20,  # Divide feature values into 20 intervals (bins)
            monte_carlo=True,  # Enable Monte Carlo simulation for robustness
            monte_carlo_rep=100,  # Set Monte Carlo repetitions to 100
            monte_carlo_ratio=0.8, # Set Monte Carlo sampling ratio to 80%
            rugplot_lim=None,
        )
        plt.show()
