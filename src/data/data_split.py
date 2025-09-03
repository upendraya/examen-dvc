import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Ensure processed folder exists
os.makedirs('data/processed', exist_ok=True)

# Load raw data
df = pd.read_csv('data/raw/raw.csv')

# Split features and target
X = df.drop('iron_feed', axis=1)  # replace 'iron_feed' with your target column if different
y = df['iron_feed']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save CSVs
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print("Data split complete. Files saved in data/processed/")
