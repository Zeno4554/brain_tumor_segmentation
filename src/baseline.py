"""Raw FLAIR intensity threshold baseline."""
import numpy as np
import pandas as pd
from .config import Config
from .data import prepare, read_image, read_mask
from .utils import segmentation_metrics

def predict_baseline(image: np.ndarray) -> np.ndarray:
    """Apply the prescribed raw, per-slice FLAIR mean-plus-three-std rule."""
    flair = image[..., 1]
    brain = flair[flair > 10]
    if not brain.size:
        return np.zeros(flair.shape, dtype=bool)
    return flair > brain.mean() + 3 * brain.std()

def run_baseline(config: Config | None = None) -> pd.DataFrame:
    config = config or Config()
    frame, splits, _ = prepare(config)
    results, details = [], []
    for patient, group in frame[frame.patient.isin(splits["test"])].groupby("patient"):
        pred = np.stack([predict_baseline(read_image(r.image)) for r in group.itertuples()])
        truth = np.stack([read_mask(r.mask) for r in group.itertuples()])
        measured = segmentation_metrics(pred, truth)
        results.append(dict(patient=patient, baseline_dice=measured["dice"]))
        details.append(dict(patient=patient, institution=patient.split("_")[1], **measured))
    table = pd.DataFrame(results)
    table.to_csv(config.output_dir / "baseline_dice.csv", index=False)
    pd.DataFrame(details).to_csv(config.output_dir / "baseline_patient_metrics.csv", index=False)
    print(table.to_string(index=False))
    return table

if __name__ == "__main__":
    run_baseline()
