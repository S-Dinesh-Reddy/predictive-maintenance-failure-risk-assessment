import json

with open('notebooks/Phase_1_Data_Preprocessing.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in Phase 1: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    ct = c['cell_type']
    lines = c.get('source', [])
    src = "".join(lines)
    first = lines[0].strip() if lines else "(empty)"
    print(f"Cell {i:02d} [{ct}]: {first[:80]}")
