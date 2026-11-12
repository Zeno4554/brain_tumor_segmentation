"""Mask-led pairing, patient splits, cached normalization, and augmentation."""
import json
import os
import random
from pathlib import Path
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, get_worker_info
from .config import Config
from .utils import write_json

def read_image(path: str | Path) -> np.ndarray:
    image = tifffile.imread(path)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected H x W x 3 MRI at {path}, got {image.shape}")
    return image.astype(np.float32)

def read_mask(path: str | Path) -> np.ndarray:
    return (tifffile.imread(path) > 0).astype(np.float32)

def discover(config: Config) -> pd.DataFrame:
    rows = []
    for mask in sorted(config.data_dir.glob("*/*_mask.tif")):
        image = Path(str(mask).replace("_mask", ""))
        assert image.is_file(), f"Missing image paired with {mask}"
        rows.append(dict(patient=mask.parent.name, index=int(mask.stem.rsplit("_", 2)[1]), image=str(image), mask=str(mask)))
    if not rows:
        raise FileNotFoundError(f"MRI data missing. Put patient folders in {Path(__file__).resolve().parents[1] / 'data/kaggle_3m'} or /kaggle/input/lgg-mri-segmentation/kaggle_3m. See scripts/download_data.md.")
    frame = pd.DataFrame(rows).sort_values(["patient", "index"]).reset_index(drop=True)
    if config.smoke_test:
        patients = frame.patient.unique()[:8]
        frame = frame[frame.patient.isin(patients)]
        frame = pd.concat([g.iloc[np.unique(np.linspace(0, len(g) - 1, min(4, len(g)), dtype=int))] for _, g in frame.groupby("patient")]).reset_index(drop=True)
    return frame

def check_leakage(splits: dict[str, list[str]]) -> None:
    sets = [set(splits[key]) for key in ("train", "val", "test")]
    assert all(sets), "Each split needs patients"
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "Patient leakage detected"

def prepare(config: Config) -> tuple[pd.DataFrame, dict, dict]:
    frame = discover(config)
    path = config.output_dir / "split.json"
    if path.exists():
        splits = json.loads(path.read_text())
        assert set(sum(splits.values(), [])) == set(frame.patient), "Saved split differs from dataset. Use a separate output directory."
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.seed)
        development, test = next(splitter.split(frame, groups=frame.patient))
        dev = frame.iloc[development]
        train, val = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=config.seed).split(dev, groups=dev.patient))
        splits = {"train": sorted(dev.iloc[train].patient.unique()), "val": sorted(dev.iloc[val].patient.unique()), "test": sorted(frame.iloc[test].patient.unique())}
    check_leakage(splits)
    write_json(path, splits)
    stats = patient_statistics(config, frame)
    return frame, splits, stats

def patient_statistics(config: Config, frame: pd.DataFrame) -> dict:
    """Cache scan-level channel statistics with the normalization mode recorded."""
    path = config.output_dir / "normalization.json"
    cached = json.loads(path.read_text()) if path.exists() else {}
    stats = {}
    for patient, group in frame.groupby("patient"):
        signature = [[r.image, Path(r.image).stat().st_size, Path(r.image).stat().st_mtime_ns] for r in group.itertuples()]
        entry = cached.get(patient, {})
        if entry.get("signature") != signature or entry.get("mode", "whole_volume") != config.normalization_mode:
            total, squared, count = np.zeros(3), np.zeros(3), np.zeros(3)
            for row in group.itertuples():
                pixels = read_image(row.image).reshape(-1, 3).astype(np.float64)
                if not np.isfinite(pixels).all():
                    raise ValueError(f"Non-finite MRI intensities in {row.image}")
                total += pixels.sum(0)
                squared += (pixels ** 2).sum(0)
                if config.normalization_mode == "foreground_nonzero":
                    count += (pixels != 0).sum(0)
                else:
                    count += len(pixels)
            support = np.maximum(count, 1)
            mean = total / support
            std = np.sqrt(np.maximum(squared / support - mean ** 2, 1e-6))
            std[count == 0] = 1
            entry = dict(mean=mean.tolist(), std=std.tolist(), signature=signature)
            if config.normalization_mode == "foreground_nonzero":
                entry.update(mode=config.normalization_mode, count=count.astype(int).tolist())
        stats[patient] = entry
    write_json(path, stats)
    return stats

def normalize(image: np.ndarray, stats: dict) -> np.ndarray:
    normalized = ((image - np.asarray(stats["mean"], dtype=np.float32)) / np.asarray(stats["std"], dtype=np.float32)).astype(np.float32)
    if stats.get("mode", "whole_volume") == "foreground_nonzero":
        normalized[image == 0] = 0
    return normalized

def seed_worker(worker_id: int) -> None:
    """Seed each worker's augmentation stream from its reproducible Torch seed."""
    info = get_worker_info()
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if info is not None and info.dataset.transform is not None:
        info.dataset.transform.set_random_seed(seed)

class MRIDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, stats: dict, augment: bool = False, seed: int = 42) -> None:
        self.frame = frame.reset_index(drop=True)
        self.stats = stats
        self.transform = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Rotate(limit=15, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=0.5)], seed=seed) if augment else None

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image, mask = normalize(read_image(row.image), self.stats[row.patient]), read_mask(row['mask'])
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
        assert np.isin(mask, [0, 1]).all()
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))), torch.from_numpy(mask[None].copy())
