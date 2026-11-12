"""Select the original checkpoint's inference threshold using validation patients."""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from src.data import prepare, read_mask
from src.evaluate import predict_probabilities
from src.model import load_model
from src.utils import device, seed_everything, write_json
from scripts.audit_model import measurements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply the validation-selected threshold to the original checkpoint")
    args = parser.parse_args()
    config = Config(prediction_threshold=0.5)
    destination = config.output_dir / "experiments/threshold_sweep"
    destination.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    frame, splits, stats = prepare(config)
    dev = device()
    model, checkpoint = load_model(config, dev)
    assert checkpoint["splits"] == splits
    assert checkpoint["normalization"] == stats
    thresholds = np.round(np.arange(0.2, 0.701, 0.05), 2)
    manifest = dict(
        thresholds=thresholds.tolist(), threshold_step=0.05,
        selection_rule="Highest mean patient-volume Dice on validation patients, ties within 1e-7 favor the threshold closest to 0.5",
        selection_patients=splits["val"], threshold_reference=0.5,
        checkpoint_sha256=hashlib.sha256((config.checkpoint_dir / "best.pt").read_bytes()).hexdigest(),
        test_patients_used_for_selection=0,
    )
    write_json(destination / "protocol.json", manifest)
    records = []
    for patient, group in frame[frame.patient.isin(splits["val"])].groupby("patient"):
        print(f"Sweeping validation patient {patient}", flush=True)
        group = group.sort_values("index")
        truth = np.stack([read_mask(row.mask) for row in group.itertuples()]).astype(bool)
        probabilities = predict_probabilities(model, group, stats, config, dev)
        for threshold in thresholds:
            records.append(dict(patient=patient, threshold=float(threshold), **measurements(probabilities >= threshold, truth)))
    table = pd.DataFrame(records)
    table.to_csv(destination / "validation_patient_metrics.csv", index=False)
    summary = table.groupby("threshold").agg(
        mean_volume_dice=("dice", "mean"), median_volume_dice=("dice", "median"),
        mean_patient_recall=("recall", "mean"), mean_patient_precision=("precision", "mean"),
        false_alarms=("false_alarms", "sum"), tumor_free_slices=("tumor_free_slices", "sum"),
        true_positive_pixels=("true_positive_pixels", "sum"), false_negative_pixels=("false_negative_pixels", "sum"),
    )
    summary["false_alarm_rate"] = summary.false_alarms / summary.tumor_free_slices
    summary["pooled_tumor_recall"] = summary.true_positive_pixels / (summary.true_positive_pixels + summary.false_negative_pixels)
    summary.to_csv(destination / "validation_summary.csv")
    np.testing.assert_allclose(summary.loc[0.5].mean_volume_dice, checkpoint["val_dice"], atol=1e-7)
    eligible = summary[summary.mean_volume_dice >= summary.mean_volume_dice.max() - 1e-7]
    selected = float(min(eligible.index, key=lambda value: abs(value - 0.5)))
    paired = table.pivot(index="patient", columns="threshold", values="dice")
    differences = (paired[selected] - paired[0.5]).to_numpy()
    rng = np.random.default_rng(config.seed)
    bootstrap = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    result = dict(
        **manifest, prediction_threshold=selected,
        validation_mean_dice=float(summary.loc[selected].mean_volume_dice),
        reference_validation_mean_dice=float(summary.loc[0.5].mean_volume_dice),
        mean_dice_change=float(differences.mean()),
        exploratory_paired_bootstrap_95_percent_interval=[float(lower), float(upper)],
        validation_patients_improved=int((differences > 1e-7).sum()),
        validation_patients_worsened=int((differences < -1e-7).sum()),
        validation_patients_unchanged=int((np.abs(differences) <= 1e-7).sum()),
        applied=args.apply,
    )
    write_json(destination / "selection.json", result)
    paired.to_csv(destination / "paired_patient_dice.csv")
    fig = Figure(figsize=(10, 3.5))
    axes = fig.subplots(1, 3)
    for ax, column, title in zip(axes, ["mean_volume_dice", "mean_patient_recall", "false_alarm_rate"], ["Mean patient volume Dice", "Mean patient tumor recall", "Tumor-free slice false-alarm rate"]):
        ax.plot(summary.index, summary[column], marker="o")
        ax.axvline(selected, color="red", linestyle="--")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Threshold")
    fig.tight_layout()
    fig.savefig(destination / "threshold_curves.png", dpi=150)
    lines = ["# Original-model threshold selection", "", "The original epoch-21 checkpoint was evaluated on all 14 validation patients at thresholds 0.20 through 0.70 in steps of 0.05. The selection rule maximizes mean patient-volume Dice. Ties within 1e-7 favor the threshold closest to 0.5. Recall and false alarms are measured as tradeoffs. Test patients contribute zero observations to threshold selection.", "", "| Threshold | Mean volume Dice | Median volume Dice | Mean patient recall | Mean patient precision | False alarms | False-alarm rate |", "|---|---:|---:|---:|---:|---:|---:|"]
    for threshold, row in summary.iterrows():
        lines.append(f"| {threshold:.2f} | {row.mean_volume_dice:.4f} | {row.median_volume_dice:.4f} | {row.mean_patient_recall:.2%} | {row.mean_patient_precision:.2%} | {int(row.false_alarms)}/{int(row.tumor_free_slices)} | {row.false_alarm_rate:.2%} |")
    lines += ["", f"Selected threshold **{selected:.2f}**. Mean validation Dice changes by {differences.mean():+.4f}. Dice improves for {result['validation_patients_improved']} patients and worsens for {result['validation_patients_worsened']}.", "", f"An exploratory paired patient-bootstrap 95% interval for the mean change is [{lower:+.4f}, {upper:+.4f}], based on 10,000 resamples with seed 42. This interval omits threshold-selection uncertainty and checkpoint-selection uncertainty. Confirmation on additional independent patients is needed to establish generalization.", "", "![Threshold comparison](threshold_curves.png)", "", "Application stores the selected threshold in `checkpoints/inference.json`, bound to the exact checkpoint SHA-256. Model loading reads that file for default evaluation and demo inference. An explicit `Config(prediction_threshold=0.5)` reproduces the original inference setting. A new checkpoint with a different fingerprint falls back to 0.5 and emits a recalibration warning.", "", f"Checkpoint SHA-256 `{manifest['checkpoint_sha256']}`."]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.apply:
        archive = config.output_dir / "reference_threshold_0_5"
        archive.mkdir(exist_ok=True)
        for name in ["metrics.json", "per_patient_dice.csv", "patient_dice.png", "demo_overlay.png", "failure_by_size.csv", "failure_by_region.csv"]:
            source = config.output_dir / name
            if source.exists() and not (archive / name).exists():
                shutil.copy2(source, archive / name)
        for name in ["04_evaluate.ipynb", "05_failures.ipynb"]:
            if not (archive / name).exists():
                shutil.copy2(ROOT / "notebooks" / name, archive / name)
        temporary = config.checkpoint_dir / "inference.tmp"
        write_json(temporary, result)
        os.replace(temporary, config.checkpoint_dir / "inference.json")
    print(summary.to_string(), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
