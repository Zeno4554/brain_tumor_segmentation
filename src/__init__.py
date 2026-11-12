"""Patient-level MRI segmentation pipeline."""
import os
from pathlib import Path

_plot_cache = Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"
_plot_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_plot_cache))
