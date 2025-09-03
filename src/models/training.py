import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

os.makedirs('models', exist_ok=True)

# Load normalized data
X_train = pd.read_csv("data/normalized/X_train.csv")
X_test = pd.read_csv("data/normalized/X_test.csv")
y_train = pd.read_csv("data/normalized/y_train.csv")
y_test = pd.read_csv("data/normalized/y_test.csv")

# Load best parameters
best_params = joblib.load('models/best_params.pkl')

# Train RandomForest with best params
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train.values.ravel())

# Save trained model
joblib.dump(model, 'models/model.joblib')

# Evaluate and print MSE
y_pred = model.predict(X_test)
mse = ((y_test.values.ravel() - y_pred) ** 2).mean()
print(f"Test MSE: {mse:.4f}")
print("Model saved to models/model.joblib")
