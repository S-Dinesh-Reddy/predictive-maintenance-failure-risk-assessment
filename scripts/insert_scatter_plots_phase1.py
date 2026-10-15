import json

notebook_path = 'notebooks/Phase_1_Data_Preprocessing.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Locate cell 30 (the boxplots) and cell 31 (Section 6 Correlation Analysis)
target_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if '## 6. Correlation Analysis' in src:
        target_idx = i
        break

assert target_idx is not None, "Could not find Section 6: Correlation Analysis in notebook"
print(f"Found Section 6 at index {target_idx}. Boxplots are at index {target_idx - 1}.")

# Create the new Markdown cell
md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Bivariate Sensor Relationships by Machine Failure\n",
        "\n",
        "Examining pairwise relationships between continuous sensor features reveals critical physical operating boundaries and failure regimes:\n",
        "- **Rotational Speed vs. Torque**: Visualizes the inverse mechanical power operating curve ($P \\propto \\text{Torque} \\times \\text{Speed}$). Failures cluster prominently along the extreme high-torque/low-speed and high-speed/low-torque boundary regimes (power failure conditions).\n",
        "- **Tool Wear vs. Torque**: Shows the coupling between cumulative cutting tool wear and mechanical cutting resistance. Failures occur disproportionately when high tool wear combines with elevated torque loads (overstrain and tool wear failure regimes)."
    ]
}

# Create Code Cell 1: Rotational Speed vs Torque
code_cell_1 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Rotational Speed vs Torque by Machine Failure\n",
        "plt.figure(figsize=(9, 6))\n",
        "plt.scatter(\n",
        "    df_clean.loc[df_clean['Machine failure'] == 0, 'Rotational speed [rpm]'],\n",
        "    df_clean.loc[df_clean['Machine failure'] == 0, 'Torque [Nm]'],\n",
        "    c='#2563eb',\n",
        "    label='No Failure (0)',\n",
        "    alpha=0.35,\n",
        "    s=25,\n",
        "    edgecolors='none'\n",
        ")\n",
        "plt.scatter(\n",
        "    df_clean.loc[df_clean['Machine failure'] == 1, 'Rotational speed [rpm]'],\n",
        "    df_clean.loc[df_clean['Machine failure'] == 1, 'Torque [Nm]'],\n",
        "    facecolors='none',\n",
        "    edgecolors='#dc2626',\n",
        "    linewidths=1.5,\n",
        "    label='Machine Failure (1)',\n",
        "    alpha=0.95,\n",
        "    s=55\n",
        ")\n",
        "\n",
        "plt.title('Rotational Speed vs Torque by Machine Failure', fontsize=13, fontweight='bold', pad=12)\n",
        "plt.xlabel('Rotational speed [rpm]', fontsize=11)\n",
        "plt.ylabel('Torque [Nm]', fontsize=11)\n",
        "plt.grid(True, linestyle='--', alpha=0.5)\n",
        "plt.legend(frameon=True, loc='upper right')\n",
        "plt.show()"
    ]
}

# Create Code Cell 2: Tool Wear vs Torque
code_cell_2 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Tool Wear vs Torque by Machine Failure\n",
        "plt.figure(figsize=(9, 6))\n",
        "plt.scatter(\n",
        "    df_clean.loc[df_clean['Machine failure'] == 0, 'Tool wear [min]'],\n",
        "    df_clean.loc[df_clean['Machine failure'] == 0, 'Torque [Nm]'],\n",
        "    c='#2563eb',\n",
        "    label='No Failure (0)',\n",
        "    alpha=0.35,\n",
        "    s=25,\n",
        "    edgecolors='none'\n",
        ")\n",
        "plt.scatter(\n",
        "    df_clean.loc[df_clean['Machine failure'] == 1, 'Tool wear [min]'],\n",
        "    df_clean.loc[df_clean['Machine failure'] == 1, 'Torque [Nm]'],\n",
        "    facecolors='none',\n",
        "    edgecolors='#dc2626',\n",
        "    linewidths=1.5,\n",
        "    label='Machine Failure (1)',\n",
        "    alpha=0.95,\n",
        "    s=55\n",
        ")\n",
        "\n",
        "plt.title('Tool Wear vs Torque by Machine Failure', fontsize=13, fontweight='bold', pad=12)\n",
        "plt.xlabel('Tool wear [min]', fontsize=11)\n",
        "plt.ylabel('Torque [Nm]', fontsize=11)\n",
        "plt.grid(True, linestyle='--', alpha=0.5)\n",
        "plt.legend(frameon=True, loc='upper right')\n",
        "plt.show()"
    ]
}

# Insert the cells at target_idx
new_cells = nb['cells'][:target_idx] + [md_cell, code_cell_1, code_cell_2] + nb['cells'][target_idx:]
nb['cells'] = new_cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Successfully inserted 3 cells. Total cells updated from {len(new_cells) - 3} to {len(new_cells)}.")
