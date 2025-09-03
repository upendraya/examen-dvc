import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os

os.makedirs('models', exist_ok=True)

# Load data
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

# Load best parameters
best_params = joblib.load('models/best_params.pkl')

# Train model
model = GradientBoostingRegressor(**best_params)
model.fit(X_train, y_train.values.ravel())

# Save trained model
joblib.dump(model, 'models/gbr_model.pkl')
