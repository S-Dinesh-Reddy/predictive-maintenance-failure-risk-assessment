import json

with open('notebooks/Phase_1_Data_Preprocessing.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('scripts/inspect_phase1_all.txt', 'w', encoding='utf-8') as out:
    for i, c in enumerate(nb['cells']):
        out.write(f"=== Cell {i:02d} [{c['cell_type']}] ===\n")
        out.write("".join(c.get('source', [])))
        out.write("\n\n")

print("Dumped Phase 1 cells to scripts/inspect_phase1_all.txt")
