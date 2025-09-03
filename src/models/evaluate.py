import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os

os.makedirs('metrics', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Load data
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

# Load trained model
model = joblib.load('models/gbr_model.pkl')

# Predict
y_pred = model.predict(X_test)
pd.DataFrame(y_pred, columns=['predictions']).to_csv('data/predictions.csv', index=False)

# Compute metrics
metrics = {
    'MSE': mean_squared_error(y_test, y_pred),
    'R2': r2_score(y_test, y_pred)
}

# Save metrics
with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f)
