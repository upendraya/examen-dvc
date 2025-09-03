import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load normalized data
X_train = pd.read_csv("data/normalized/X_train.csv")
X_test = pd.read_csv("data/normalized/X_test.csv")
y_train = pd.read_csv("data/normalized/y_train.csv")
y_test = pd.read_csv("data/normalized/y_test.csv")

# Ensure target is 1D
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("Model trained and saved to models/model.pkl")
