import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Ensure models folder exists
os.makedirs('models', exist_ok=True)

# Load normalized data
X_train = pd.read_csv("data/normalized/X_train.csv")
X_test = pd.read_csv("data/normalized/X_test.csv")
y_train = pd.read_csv("data/normalized/y_train.csv")
y_test = pd.read_csv("data/normalized/y_test.csv")

# Train RandomForest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())  # y_train may need .values.ravel() for 1D

# Save model
joblib.dump(model, 'models/rf_model.pkl')

print("Training complete. Model saved in models/rf_model.pkl")
