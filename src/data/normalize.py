import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Ensure normalized folder exists
os.makedirs("data/normalized", exist_ok=True)

# Load processed data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Drop non-numeric columns
numeric_cols = X_train.select_dtypes(include="number").columns
X_train_numeric = X_train[numeric_cols]
X_test_numeric = X_test[numeric_cols]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_numeric)
X_test_scaled = scaler.transform(X_test_numeric)

# Save normalized data
pd.DataFrame(X_train_scaled, columns=numeric_cols).to_csv("data/normalized/X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=numeric_cols).to_csv("data/normalized/X_test.csv", index=False)
y_train.to_csv("data/normalized/y_train.csv", index=False)
y_test.to_csv("data/normalized/y_test.csv", index=False)

print("Normalization complete. Files saved in data/normalized/")
