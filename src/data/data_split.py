import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Ensure processed folder exists
os.makedirs("data/processed", exist_ok=True)

# Load raw data
df = pd.read_csv("data/raw/raw.csv")

# Features (drop target)
X = df.drop(columns=["silica_concentrate"])
# Target
y = df[["silica_concentrate"]]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save to processed
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Data split complete. Files saved in data/processed/")
