import json

with open(r'C:\Users\Dinesh\Downloads\Phase_2_Model_Training_Evaluation (3).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
md_cells = [(i, "".join(c['source'])) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'markdown']
print(f"Markdown cells count: {len(md_cells)}")
for idx, text in md_cells:
    first = text.strip().split('\n')[0]
    print(f"MD Cell {idx:03d}: {first[:80]}")
