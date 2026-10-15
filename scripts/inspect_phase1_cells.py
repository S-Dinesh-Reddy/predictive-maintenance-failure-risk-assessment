import json

with open('notebooks/Phase_1_Data_Preprocessing.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in range(29, 36):
    cell = nb['cells'][i]
    src = ''.join(cell['source'])
    print(f"=== Cell {i} ({cell['cell_type']}) ===")
    print(src[:250])
    print("-" * 50)
