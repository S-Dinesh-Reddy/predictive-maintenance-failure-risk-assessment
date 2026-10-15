import json

with open('notebooks/Phase_1_Data_Preprocessing.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")

error_cells = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'error':
                error_cells.append((i, output.get('ename'), output.get('evalue')))

print(f"Cells with errors: {len(error_cells)}")
if error_cells:
    for idx, name, val in error_cells:
        print(f"  Cell {idx}: {name} - {val}")

# Check outputs of Cell 32 and 33 specifically
for idx in [32, 33]:
    cell = nb['cells'][idx]
    first_line = ''.join(cell['source']).split('\n')[0]
    out_types = [out.get('output_type') for out in cell.get('outputs', [])]
    has_image = any('image/png' in out.get('data', {}) for out in cell.get('outputs', []) if out.get('output_type') == 'display_data')
    print(f"Cell {idx} ({first_line}):")
    print(f"  Execution count: {cell.get('execution_count')}")
    print(f"  Output types: {out_types}")
    print(f"  Contains PNG image: {has_image}")
