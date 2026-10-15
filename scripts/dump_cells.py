import json

with open(r'C:\Users\Dinesh\Downloads\Phase_2_Model_Training_Evaluation (3).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
with open(r'scripts/inspect_all_cells.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        ct = cell['cell_type']
        src = "".join(cell.get('source', []))
        first_line = cell['source'][0].strip() if cell.get('source') else ""
        out.write(f"=== CELL {i:03d} [{ct}] ===\n")
        out.write(f"First line: {first_line}\n")
        out.write(f"Length lines: {len(cell.get('source', []))}\n")
        out.write(src[:300] + ("..." if len(src) > 300 else "") + "\n\n")

print("Wrote inspect_all_cells.txt")
