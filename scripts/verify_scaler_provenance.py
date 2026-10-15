import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load original dataset
df = pd.read_csv('data/ai4i2020.csv')

# Replicate Phase 1 steps exactly
df_clean = df.copy()
df_clean = df_clean.drop(columns=[
    'UDI',
    'Product ID',
    'TWF',
    'HDF',
    'PWF',
    'OSF',
    'RNF'
])
df_clean = pd.get_dummies(
    df_clean,
    columns=['Type'],
    drop_first=True,
    dtype=int
)
df_clean['Temperature difference [K]'] = (
    df_clean['Process temperature [K]'] - df_clean['Air temperature [K]']
)
df_clean['Tool wear level'] = pd.qcut(
    df_clean['Tool wear [min]'],
    q=3,
    labels=['Low', 'Medium', 'High']
).map({
    'Low': 0,
    'Medium': 1,
    'High': 2
}).astype(int)

X = df_clean.drop(columns=['Machine failure'])
y = df_clean['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare with data/X_train_scaled.csv
X_train_file = pd.read_csv('data/X_train_scaled.csv')
X_test_file = pd.read_csv('data/X_test_scaled.csv')

max_diff_train = np.max(np.abs(X_train_scaled - X_train_file.values))
max_diff_test = np.max(np.abs(X_test_scaled - X_test_file.values))

print("Feature Names in Order:", list(X.columns))
print("Sample Count Seen     :", scaler.n_samples_seen_)
print("Scaler Means          :", list(np.round(scaler.mean_, 6)))
print("Scaler Scales (Std)   :", list(np.round(scaler.scale_, 6)))
print("Scaler Variances      :", list(np.round(scaler.var_, 6)))
print("Max Diff vs X_train_scaled.csv:", max_diff_train)
print("Max Diff vs X_test_scaled.csv :", max_diff_test)

# Also check saved models/scaler.pkl
saved_scaler = joblib.load('models/scaler.pkl')
mean_diff = np.max(np.abs(scaler.mean_ - saved_scaler.mean_))
scale_diff = np.max(np.abs(scaler.scale_ - saved_scaler.scale_))
print("Max Diff vs models/scaler.pkl Mean :", mean_diff)
print("Max Diff vs models/scaler.pkl Scale:", scale_diff)
