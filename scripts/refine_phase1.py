import json
import os

with open('notebooks/Phase_1_Data_Preprocessing.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Let's inspect the original cells and refine them
print(f"Original cell count: {len(nb['cells'])}")
