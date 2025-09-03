import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

os.makedirs('models', exist_ok=True)

# Load data
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

# Model and parameters
model = GradientBoostingRegressor()
params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

grid = GridSearchCV(model, param_grid=params, cv=5, scoring='r2')
grid.fit(X_train, y_train.values.ravel())

# Save best parameters
joblib.dump(grid.best_params_, 'models/best_params.pkl')
