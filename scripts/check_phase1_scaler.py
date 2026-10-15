import json
import pandas as pd
import numpy as np

with open('notebooks/Phase_1_Data_Preprocessing.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in range(35, 50):
    c = nb['cells'][i]
    src = "".join(c.get('source', []))
    print(f"=== Cell {i} [{c['cell_type']}] ===")
    print(src[:400])
