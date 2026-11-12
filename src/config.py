"""Portable configuration shared by notebooks and command-line tools."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 8
ENCODER = "resnet34"
EPOCHS = 30
NUM_WORKERS = 2
SMOKE_TEST = False

@dataclass
class Config:
    data_dir: Path | None = None
    output_dir: Path = ROOT / "outputs"
    checkpoint_dir: Path = ROOT / "checkpoints"
    batch_size: int = BATCH_SIZE
    encoder: str = ENCODER
    epochs: int = EPOCHS
    num_workers: int = NUM_WORKERS
    smoke_test: bool = SMOKE_TEST
    seed: int = 42
    encoder_weights: str | None = "imagenet"
    target_positive_fraction: float | None = None
    prediction_threshold: float | None = None
    normalization_mode: str = "whole_volume"
    loss_name: str = "dice_bce"

    def __post_init__(self) -> None:
        if self.loss_name not in {"dice_bce", "tversky_bce"}:
            raise ValueError("loss_name must be dice_bce or tversky_bce")
        if self.normalization_mode not in {"whole_volume", "foreground_nonzero"}:
            raise ValueError("normalization_mode must be whole_volume or foreground_nonzero")
        if self.prediction_threshold is not None and not 0 < self.prediction_threshold < 1:
            raise ValueError("prediction_threshold must be between 0 and 1")
        if self.target_positive_fraction is not None and not 0 < self.target_positive_fraction < 1:
            raise ValueError("target_positive_fraction must be between 0 and 1")
        if self.data_dir is None:
            local = ROOT / "data/kaggle_3m"
            self.data_dir = local if local.exists() else Path("/kaggle/input/lgg-mri-segmentation/kaggle_3m")
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        if self.smoke_test:
            if self.output_dir == ROOT / "outputs":
                self.output_dir = self.output_dir / "smoke"
            if self.checkpoint_dir == ROOT / "checkpoints":
                self.checkpoint_dir = self.checkpoint_dir / "smoke"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.smoke_test:
            self.epochs = min(self.epochs, 2)
            self.num_workers = 0
