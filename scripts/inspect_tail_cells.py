import json

with open('notebooks/Phase_2_Model_Training_Evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in range(68, len(nb['cells'])):
    c = nb['cells'][i]
    src = "".join(c.get('source', []))
    first = c['source'][0].strip() if c.get('source') else "(empty)"
    print(f"Cell {i:03d} [{c['cell_type'][:4]}]: {first[:80]}")
