"""Reproducibility, device selection, and volume metrics."""
import json
import random
import warnings
from pathlib import Path
import numpy as np
import torch

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(min(4, torch.get_num_threads()))

def device() -> torch.device:
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    warnings.warn("CUDA unavailable. Falling back to CPU.", stacklevel=2)
    return torch.device("cpu")

def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    """Compute one Dice across the complete volume, including empty volumes."""
    if pred.shape != truth.shape:
        raise ValueError("Prediction and annotation must have the same shape")
    pred, truth = pred.astype(bool), truth.astype(bool)
    denominator = int(pred.sum()) + int(truth.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(pred, truth).sum() / (denominator + 1e-7))

def segmentation_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Measure overlap, pixel errors, and positive/negative slice failures."""
    score = dice(pred, truth)
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    positive = truth.any(axis=(-2, -1))
    predicted = pred.any(axis=(-2, -1))
    tp_per_slice = (pred & truth).sum(axis=(-2, -1))
    fp_per_slice = (pred & ~truth).sum(axis=(-2, -1))
    negative_fp = int(np.sum(np.where(~positive, fp_per_slice, 0)))
    return dict(
        dice=score, precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        true_positive_pixels=tp, false_positive_pixels=fp, false_negative_pixels=fn,
        predicted_pixels=tp + fp, annotated_pixels=tp + fn,
        false_alarms=int(np.sum(~positive & predicted)), tumor_free_slices=int(np.sum(~positive)),
        tumor_containing_slices=int(np.sum(positive)),
        completely_missed_positive_slices=int(np.sum(positive & ~predicted)),
        zero_overlap_positive_slices=int(np.sum(positive & (tp_per_slice == 0))),
        false_positive_pixels_on_tumor_free_slices=negative_fp,
        false_positive_pixels_on_positive_slices=fp - negative_fp,
    )

def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
