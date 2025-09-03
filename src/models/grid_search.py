import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

# Ensure models folder exists
os.makedirs('models', exist_ok=True)

# Load normalized training data
X_train = pd.read_csv("data/normalized/X_train.csv")
y_train = pd.read_csv("data/normalized/y_train.csv")

# Define model
rf = RandomForestRegressor(random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Grid search
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train.values.ravel())

# Save best parameters
joblib.dump(grid_search.best_params_, 'models/best_params.pkl')
print("Grid search complete. Best parameters saved in models/best_params.pkl")

