"""Execute project notebooks and retain outputs after each cell."""
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
for variable, folder in {
    "TORCH_HOME": "torch_cache",
    "HF_HOME": "huggingface_cache",
    "IPYTHONDIR": ".ipython",
    "JUPYTER_RUNTIME_DIR": ".jupyter_runtime",
}.items():
    os.environ.setdefault(variable, str(root / "outputs" / folder))
os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

for path in sorted((root / "notebooks").glob("*.ipynb")):
    if sys.argv[1:] and path.stem not in sys.argv[1:]:
        continue
    print(f"Running {path.name}", flush=True)
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    )
    client.on_cell_executed = lambda **kwargs: nbformat.write(notebook, path)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    print(f"Completed {path.name}", flush=True)
