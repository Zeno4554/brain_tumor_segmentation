"""Held-out volume inference, metrics, and plots."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .baseline import run_baseline
from .config import Config
from .data import MRIDataset, prepare, read_mask
from .model import load_model
from .utils import device, segmentation_metrics, write_json

@torch.no_grad()
def predict_probabilities(model: torch.nn.Module, frame: pd.DataFrame, stats: dict, config: Config, dev: torch.device) -> np.ndarray:
    """Return numerically ordered slice probabilities for one patient scan."""
    loader = DataLoader(MRIDataset(frame.sort_values("index"), stats), batch_size=config.batch_size, num_workers=0)
    predictions = []
    model.eval()
    for images, _ in loader:
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            predictions.append(model(images.to(dev)).sigmoid().float().cpu().numpy()[:, 0])
    return np.concatenate(predictions)

def predict_patient(model: torch.nn.Module, frame: pd.DataFrame, stats: dict, config: Config, dev: torch.device) -> np.ndarray:
    """Threshold one scan using the checkpoint's inference configuration."""
    threshold = config.prediction_threshold if config.prediction_threshold is not None else 0.5
    return predict_probabilities(model, frame, stats, config, dev) >= threshold

def evaluate(config: Config | None = None) -> pd.DataFrame:
    config = config or Config()
    dev = device()
    model, checkpoint = load_model(config, dev)
    frame, splits, stats = prepare(config)
    assert checkpoint["splits"] == splits, "Checkpoint patient split mismatch"
    assert checkpoint["normalization"] == stats, "Checkpoint normalization mismatch"
    baseline = run_baseline(config)
    rows, slice_rows = [], []
    for patient, group in frame[frame.patient.isin(splits["test"])].groupby("patient"):
        group = group.sort_values("index")
        pred = predict_patient(model, group, stats, config, dev)
        truth = np.stack([read_mask(r.mask) for r in group.itertuples()])
        for position, row in enumerate(group.itertuples()):
            slice_rows.append(dict(patient=patient, institution=patient.split("_")[1], index=row.index,
                                   **segmentation_metrics(pred[position], truth[position])))
        measured = segmentation_metrics(pred, truth)
        score = measured.pop("dice")
        rows.append(dict(patient=patient, institution=patient.split("_")[1], unet_dice=score, tumor_voxels=int(truth.sum()), **measured))
    table = pd.DataFrame(rows).merge(baseline, on="patient", validate="one_to_one")
    table.to_csv(config.output_dir / "per_patient_dice.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(config.output_dir / "per_slice_errors.csv", index=False)
    metrics = {f"{name}_{stat}": float(getattr(table[f"{name}_dice"], stat)()) for name in ("unet", "baseline") for stat in ("mean", "median")}
    alarms, empty = int(table.false_alarms.sum()), int(table.tumor_free_slices.sum())
    tp, fp, fn = (int(table[column].sum()) for column in ("true_positive_pixels", "false_positive_pixels", "false_negative_pixels"))
    metrics.update(
        false_alarms=alarms, tumor_free_slices=empty, false_alarm_rate=alarms / empty if empty else None,
        test_patients=len(table), smoke_test=config.smoke_test, reference_mean_dice=0.82,
        prediction_threshold=config.prediction_threshold, normalization_mode=config.normalization_mode,
        mean_patient_precision=float(table.precision.mean()), mean_patient_recall=float(table.recall.mean()),
        pooled_precision=tp / (tp + fp) if tp + fp else None, pooled_recall=tp / (tp + fn) if tp + fn else None,
        true_positive_pixels=tp, false_positive_pixels=fp, false_negative_pixels=fn,
        completely_missed_positive_slices=int(table.completely_missed_positive_slices.sum()),
        zero_overlap_positive_slices=int(table.zero_overlap_positive_slices.sum()),
        completely_missed_slice_definition="annotated slice with no predicted positive pixels anywhere",
        zero_overlap_slice_definition="annotated slice with zero true-positive pixels, including off-target predictions",
        tumor_containing_slices=int(table.tumor_containing_slices.sum()),
        false_positive_pixels_on_tumor_free_slices=int(table.false_positive_pixels_on_tumor_free_slices.sum()),
        false_positive_pixels_on_positive_slices=int(table.false_positive_pixels_on_positive_slices.sum()),
        checkpoint_sha256=checkpoint["checkpoint_sha256"], checkpoint_epoch=int(checkpoint["epoch"]),
        checkpoint_validation_dice_at_0_5=float(checkpoint["val_dice"]),
        augmentation_seeding=checkpoint.get("augmentation_seeding", "legacy_fixed_transform_seed"),
        evaluation_role="descriptive remeasurement of the fixed deployed checkpoint; test cases previously inspected",
    )
    write_json(config.output_dir / "metrics.json", metrics)
    deployment = {key: metrics[key] for key in ("checkpoint_sha256", "checkpoint_epoch", "augmentation_seeding", "normalization_mode", "prediction_threshold")}
    deployment.update(checkpoint_file=str(config.checkpoint_dir / "best.pt"), metrics_file=str(config.output_dir / "metrics.json"))
    write_json(config.output_dir / "deployment_manifest.json", deployment)
    institution_metrics(table).to_csv(config.output_dir / "per_institution_metrics.csv", index=False)
    print(table[["patient", "unet_dice", "precision", "recall", "false_alarms", "completely_missed_positive_slices", "baseline_dice"]].to_string(index=False))
    print("\nHeld-out patient summary")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    if metrics["unet_mean"] > 0.90:
        print("WARNING: mean Dice exceeds 0.90. Re-check patient leakage. Buda et al. reference is about 0.82.")
    plot_results(config)
    from .app import save_sample
    sample = max(
        frame[frame.patient == splits["test"][0]].itertuples(),
        key=lambda row: read_mask(row.mask).sum(),
    )
    save_sample(config, sample.image, sample.patient)
    return table

def institution_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize patient scores and pooled errors by source institution code."""
    result = table.groupby("institution").agg(
        patients=("patient", "size"), mean_volume_dice=("unet_dice", "mean"), median_volume_dice=("unet_dice", "median"),
        mean_patient_precision=("precision", "mean"), mean_patient_recall=("recall", "mean"),
        true_positive_pixels=("true_positive_pixels", "sum"), false_positive_pixels=("false_positive_pixels", "sum"),
        false_negative_pixels=("false_negative_pixels", "sum"), false_alarms=("false_alarms", "sum"),
        tumor_free_slices=("tumor_free_slices", "sum"), tumor_containing_slices=("tumor_containing_slices", "sum"),
        completely_missed_positive_slices=("completely_missed_positive_slices", "sum"),
    )
    if "zero_overlap_positive_slices" in table:
        result["zero_overlap_positive_slices"] = table.groupby("institution").zero_overlap_positive_slices.sum()
    result["false_alarm_rate"] = result.false_alarms / result.tumor_free_slices.replace(0, np.nan)
    result["pooled_precision"] = result.true_positive_pixels / (result.true_positive_pixels + result.false_positive_pixels).replace(0, np.nan)
    result["pooled_recall"] = result.true_positive_pixels / (result.true_positive_pixels + result.false_negative_pixels).replace(0, np.nan)
    return result.reset_index()

def plot_results(config: Config) -> None:
    from matplotlib.figure import Figure
    table = pd.read_csv(config.output_dir / "per_patient_dice.csv")
    fig = Figure(figsize=(8, 4))
    ax = fig.subplots()
    x = np.arange(len(table))
    ax.bar(x - 0.2, table.unet_dice, 0.4, label="U-Net")
    ax.bar(x + 0.2, table.baseline_dice, 0.4, label="FLAIR baseline")
    ax.set_xticks(x, table.patient, rotation=90, fontsize=7)
    ax.legend()
    ax.set_ylabel("Volume Dice")
    fig.tight_layout()
    fig.savefig(config.output_dir / "patient_dice.png")
    history = pd.read_csv(config.output_dir / "history.csv")
    fig = Figure(figsize=(9, 3))
    axes = fig.subplots(1, 2)
    axes[0].plot(history.epoch, history.train_loss)
    axes[0].set_title("Training loss")
    axes[1].plot(history.epoch, history.val_dice)
    axes[1].set_title("Validation volume Dice")
    fig.tight_layout()
    fig.savefig(config.output_dir / "training_curves.png")
    institutions = pd.read_csv(config.output_dir / "per_institution_metrics.csv")
    fig = Figure(figsize=(7, 3.5))
    ax = fig.subplots()
    ax.bar(institutions.institution, institutions.mean_volume_dice)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean patient-volume Dice")
    ax.set_xlabel("Institution code")
    for position, row in enumerate(institutions.itertuples()):
        ax.text(position, row.mean_volume_dice + 0.02, f"n={row.patients}", ha="center")
    fig.tight_layout()
    fig.savefig(config.output_dir / "per_institution_dice.png")

if __name__ == "__main__":
    evaluate()
