"""Maintainer helper: rebuild the notebook from custom_llm.py without executing it."""
import hashlib
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parent
cells, lines, kind = [], [], None


def flush():
    if kind is None:
        return
    text = "\n".join(lines).strip()
    if kind == "markdown":
        text = "\n".join(line[2:] if line.startswith("# ") else line[1:]
                         if line.startswith("#") else line for line in lines).strip()
        cell = nbformat.v4.new_markdown_cell(text)
    else:
        cell = nbformat.v4.new_code_cell(text)
    cell.id = hashlib.sha256((kind + text).encode()).hexdigest()[:12]
    cells.append(cell)


for line in (root / "custom_llm.py").read_text().splitlines():
    if line.startswith("# %%"):
        flush()
        kind, lines = ("markdown" if "[markdown]" in line else "code"), []
    else:
        lines.append(line)
flush()
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec":{"display_name":"Python 3", "language":"python", "name":"python3"},
    "language_info":{"name":"python", "version":"3.13"},
    "colab":{"name":"custom_llm.ipynb", "provenance":[]}})
nbformat.validate(notebook)
nbformat.write(notebook, root / "custom_llm.ipynb")
print("Built", len(cells), "cells")
