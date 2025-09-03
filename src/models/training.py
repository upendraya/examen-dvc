import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os

# Ensure models folder exists
os.makedirs('models', exist_ok=True)

# Load normalized data
X_train = pd.read_csv('data/normalized/X_train.csv')
X_test = pd.read_csv('data/normalized/X_test.csv')
y_train = pd.read_csv('data/normalized/y_train.csv')
y_test = pd.read_csv('data/normalized/y_test.csv')

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train.values.ravel())  # ensure correct shape

# Save model
import joblib
joblib.dump(model, 'models/model.pkl')

print("Model training complete. Model saved in models/model.pkl")

