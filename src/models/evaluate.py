import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import os

# Load normalized test data
X_test = pd.read_csv("data/normalized/X_test.csv")
y_test = pd.read_csv("data/normalized/y_test.csv").values.ravel()

# Load trained model
model = joblib.load("models/model.pkl")

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save metrics
os.makedirs("metrics", exist_ok=True)
with open("metrics/scores.txt", "w") as f:
    f.write(f"Mean Squared Error: {mse}\n")
    f.write(f"R2 Score: {r2}\n")

print("Evaluation complete. Metrics saved in metrics/scores.txt")
