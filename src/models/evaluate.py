import pandas as pd
import joblib
import os
import json

os.makedirs('data/predictions', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

# Load data and model
X_test = pd.read_csv("data/normalized/X_test.csv")
y_test = pd.read_csv("data/normalized/y_test.csv")
model = joblib.load('models/model.joblib')

# Predictions
y_pred = model.predict(X_test)
pred_df = pd.DataFrame({'y_true': y_test.values.ravel(), 'y_pred': y_pred})
pred_df.to_csv("data/predictions/predictions.csv", index=False)

# Metrics
mse = ((y_test.values.ravel() - y_pred) ** 2).mean()
r2 = model.score(X_test, y_test)
metrics = {"MSE": mse, "R2": r2}

with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f)

print("Evaluation complete. Predictions and metrics saved.")

