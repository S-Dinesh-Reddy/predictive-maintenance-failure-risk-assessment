import json

def dump_nb(nb_path, out_path):
    nb = json.load(open(nb_path, 'r', encoding='utf-8'))
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, c in enumerate(nb['cells']):
            f.write(f"=== CELL {i:02d} [{c['cell_type']}] ===\n")
            f.write("".join(c.get('source', [])))
            f.write("\n\n")
    print(f"Dumped {len(nb['cells'])} cells from {nb_path} to {out_path}")

if __name__ == '__main__':
    dump_nb('notebooks/Phase_1_Data_Preprocessing.ipynb', 'scripts/phase1_cells.txt')
    dump_nb('notebooks/Phase_2_Model_Training_Evaluation.ipynb', 'scripts/phase2_cells.txt')
