import json

with open('notebooks/Phase_2_Model_Training_Evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = "".join(c.get('source', []))
    first = c['source'][0].strip() if c.get('source') else "(empty)"
    if c['cell_type'] == 'markdown':
        print(f"Cell {i:03d} [MD]  : {first[:80]}")
    elif any(k in src for k in ['RandomizedSearchCV', 'threshold', 'cross_val_predict', 'best_xgb_model', 'leaderboard', 'Pipeline', 'StandardScaler']):
        print(f"Cell {i:03d} [CODE]: {first[:80]}")
