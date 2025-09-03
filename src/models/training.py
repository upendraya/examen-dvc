# src/models/training.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load normalized features
X_train = pd.read_csv("data/normalized/X_train_scaled.csv")
X_test = pd.read_csv("data/normalized/X_test_scaled.csv")

# Load target labels
y_train = pd.read_csv("data/processed/y_train.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# If y_train/y_test have a single column, convert to 1D arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "models/model.pkl")

print("Training complete. Model saved to models/model.pkl")


