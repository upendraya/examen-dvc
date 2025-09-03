import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Paths
normalized_path = "data/normalized"
model_path = "models"

os.makedirs(model_path, exist_ok=True)

X_train_file = os.path.join(normalized_path, "X_train.csv")
X_test_file  = os.path.join(normalized_path, "X_test.csv")
y_train_file = os.path.join(normalized_path, "y_train.csv")
y_test_file  = os.path.join(normalized_path, "y_test.csv")

# Load data
X_train = pd.read_csv(X_train_file)
X_test  = pd.read_csv(X_test_file)
y_train = pd.read_csv(y_train_file)
y_test  = pd.read_csv(y_test_file)

# Train a model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())

# Predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")

# Save model
model_file = os.path.join(model_path, "random_forest_model.pkl")
joblib.dump(model, model_file)

print(f"Model saved to {model_file}")
