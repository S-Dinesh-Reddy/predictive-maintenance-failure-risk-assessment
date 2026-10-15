import json

with open(r'C:\Users\Dinesh\Downloads\Phase_2_Model_Training_Evaluation (3).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    ct = cell['cell_type']
    lines = cell.get('source', [])
    src = "".join(lines).strip()
    first_line = lines[0].strip() if lines else "(empty)"
    
    if ct == 'markdown' and first_line.startswith('#'):
        print(f"Cell {i:03d} [MD] : {first_line[:90]}")
    elif ct == 'code':
        # keywords
        kw = [k for k in ['import', 'LogisticRegression', 'RandomForest', 'SVC', 'DecisionTree', 'XGB', 'StandardScaler', 'PCA', 'cross_val', 'confusion_matrix', 'classification_report', 'roc_curve', 'GridSearch', 'RandomizedSearch', 'pickle', 'joblib'] if k.lower() in src.lower()]
        if kw:
            print(f"Cell {i:03d} [CODE] ({', '.join(kw[:3])}): {first_line[:75]}")
