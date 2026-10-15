import pandas as pd
import numpy as np

# Load X_train_scaled
df_train = pd.read_csv('data/X_train_scaled.csv')
print("Columns:", list(df_train.columns))

# For each column, let's analyze unique values and spacing
for col in df_train.columns:
    vals = np.sort(df_train[col].unique())
    diffs = np.diff(vals)
    min_pos_diff = np.min(diffs[diffs > 1e-6])
    print(f"\nFeature: {col}")
    print(f"  Min scaled value: {vals[0]:.6f}, Max: {vals[-1]:.6f}, N unique: {len(vals)}")
    print(f"  Smallest non-zero difference between sorted unique values: {min_pos_diff:.8f}")

# Check Phase 1 cell 20 describe outputs
import json
with open('notebooks/Phase_1_Data_Preprocessing.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
c20 = nb['cells'][20]
print("\nPhase 1 Cell 20 outputs (df_clean.describe()):")
for o in c20.get('outputs', []):
    if 'text' in o:
        print(''.join(o['text']))
    elif 'data' in o and 'text/plain' in o['data']:
        print(''.join(o['data']['text/plain']))
