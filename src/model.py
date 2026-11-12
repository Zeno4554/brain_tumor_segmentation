"""U-Net construction and checkpoint loading."""
import hashlib
import json
import warnings
import segmentation_models_pytorch as smp
import torch
from .config import Config

def build_model(config: Config, pretrained: bool = True) -> torch.nn.Module:
    return smp.Unet(config.encoder, encoder_weights=config.encoder_weights if pretrained else None, in_channels=3, classes=1)

def load_model(config: Config, device: torch.device) -> tuple[torch.nn.Module, dict]:
    path = config.checkpoint_dir / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"Train first. Checkpoint missing at {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    checkpoint_fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    config.encoder = checkpoint["encoder"]
    config.normalization_mode = checkpoint.get("normalization_mode", "whole_volume")
    model = build_model(config, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if config.prediction_threshold is None:
        config.prediction_threshold = 0.5
        calibration_path = config.checkpoint_dir / "inference.json"
        if calibration_path.exists():
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            if calibration["checkpoint_sha256"] == checkpoint_fingerprint:
                threshold = float(calibration["prediction_threshold"])
                if not 0 < threshold < 1:
                    raise ValueError(f"Invalid prediction threshold in {calibration_path}")
                config.prediction_threshold = threshold
            else:
                warnings.warn("Inference calibration belongs to a different checkpoint. Using threshold 0.5. Re-run threshold selection for the new checkpoint.", stacklevel=2)
    checkpoint["prediction_threshold"] = config.prediction_threshold
    checkpoint["checkpoint_sha256"] = checkpoint_fingerprint
    return model, checkpoint
