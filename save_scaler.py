"""
Script to extract and save the StandardScaler used during model training.
Run this script in your notebook environment to save the scaler for the Streamlit app.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load and preprocess the data (same as in your notebook)
df = pd.read_csv("Medical Insurance cost prediction.csv")

# Remove duplicates
df = df.drop_duplicates()

# Log transform charges
df['charges'] = np.log(df['charges'])

# One-hot encode categorical features (same as training)
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Separate features and target
X = df.drop('charges', axis=1)
y = df['charges']

# Create and fit the scaler on the same data used for training
scaler = StandardScaler()
scaler.fit(X)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved successfully as 'scaler.pkl'")
print(f"Feature names: {list(X.columns)}")
print(f"Mean values: {scaler.mean_}")
print(f"Std values: {scaler.scale_}")
