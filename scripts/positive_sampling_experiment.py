"""Run one controlled positive-sampling experiment and compare validation volumes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "outputs/huggingface_cache"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "outputs/torch_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

def compare(reference: Config, candidate: Config, manifest: dict) -> None:
    """Compare checkpoints on all original validation patients at threshold 0.5."""
    import numpy as np
    import pandas as pd
    from src.data import prepare, read_image, read_mask
    from src.evaluate import predict_patient
    from src.model import load_model
    from src.utils import device, seed_everything, write_json
    from scripts.audit_model import measurements, save_case

    seed_everything(reference.seed)
    frame, splits, stats = prepare(reference)
    dev = device()
    models = {}
    reference_validation_score = None
    for name, config in (("reference", reference), ("candidate", candidate)):
        model, checkpoint = load_model(config, dev)
        assert checkpoint["splits"] == splits
        assert checkpoint["normalization"] == stats
        models[name] = model
        if name == "reference":
            reference_validation_score = checkpoint["val_dice"]
    records = []
    for patient, group in frame[frame.patient.isin(splits["val"])].groupby("patient"):
        print(f"Comparing validation patient {patient}", flush=True)
        group = group.sort_values("index")
        truth = np.stack([read_mask(row.mask) for row in group.itertuples()]).astype(bool)
        for name, model in models.items():
            pred = predict_patient(model, group, stats, reference, dev)
            records.append(dict(patient=patient, model=name, **measurements(pred, truth)))
            worst = int(np.argmax(np.logical_xor(pred, truth).sum(axis=(1, 2))))
            row = group.iloc[worst]
            save_case(candidate.output_dir / f"{patient}_{name}_worst_slice.png", read_image(row.image), truth[worst], pred[worst], f"{name}, {patient}, slice {row['index']}")
    table = pd.DataFrame(records)
    table.to_csv(candidate.output_dir / "validation_patient_metrics.csv", index=False)
    summary = table.groupby("model").agg(
        mean_volume_dice=("dice", "mean"), median_volume_dice=("dice", "median"),
        macro_tumor_recall=("recall", "mean"), macro_precision=("precision", "mean"),
        false_alarms=("false_alarms", "sum"), tumor_free_slices=("tumor_free_slices", "sum"),
        true_positive_pixels=("true_positive_pixels", "sum"), false_negative_pixels=("false_negative_pixels", "sum"),
    )
    summary["false_alarm_rate"] = summary.false_alarms / summary.tumor_free_slices
    summary["pooled_tumor_recall"] = summary.true_positive_pixels / (summary.true_positive_pixels + summary.false_negative_pixels)
    summary.to_csv(candidate.output_dir / "validation_summary.csv")
    paired = table.pivot(index="patient", columns="model", values="dice")
    paired["dice_change"] = paired.candidate - paired.reference
    paired.to_csv(candidate.output_dir / "paired_dice.csv")
    differences = paired.dice_change.to_numpy()
    rng = np.random.default_rng(reference.seed)
    samples = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    result = dict(
        mean_dice_change=float(differences.mean()),
        paired_bootstrap_95_percent_interval=[float(lower), float(upper)],
        patients_improved=int((differences > 1e-7).sum()),
        patients_worsened=int((differences < -1e-7).sum()),
        patients_unchanged=int((np.abs(differences) <= 1e-7).sum()),
        validation_patients=len(paired), test_patients_evaluated=0,
        reference_checkpoint_sha256=hashlib.sha256((reference.checkpoint_dir / "best.pt").read_bytes()).hexdigest(),
        candidate_checkpoint_sha256=hashlib.sha256((candidate.checkpoint_dir / "best.pt").read_bytes()).hexdigest(),
    )
    assert result["reference_checkpoint_sha256"] == manifest["reference_checkpoint_sha256"]
    np.testing.assert_allclose(summary.loc["reference", "mean_volume_dice"], reference_validation_score, atol=1e-7)
    write_json(candidate.output_dir / "comparison.json", result)
    lines = [
        "# Positive-sampling experiment", "",
        "This experiment changes training slice sampling to an expected 50% tumor-containing slices and 50% tumor-free slices. Weighted sampling uses replacement and retains 2,710 training draws per epoch. The target fraction is an expectation, so actual epoch proportions vary. Original sampling used each training slice once per shuffled epoch, with 33.43% positive slices.", "",
        "Both runs use seed 42, ImageNet-initialized ResNet34 U-Net, 30 epochs, batch size 8, the original loss, augmentations, learning rates, scheduler, and the fixed 74/14/22 patient split. Each checkpoint is selected by its own best mean validation-volume Dice. The sampling choice also changes which slices recur and their augmentation exposure.", "",
        "## Validation results", "",
        "| Model | Mean volume Dice | Median volume Dice | Mean patient recall | Pooled recall | False alarms | False-alarm rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary.iterrows():
        lines.append(f"| {name} | {row.mean_volume_dice:.4f} | {row.median_volume_dice:.4f} | {row.macro_tumor_recall:.2%} | {row.pooled_tumor_recall:.2%} | {int(row.false_alarms)}/{int(row.tumor_free_slices)} | {row.false_alarm_rate:.2%} |")
    lines += ["", f"Mean paired Dice change is {differences.mean():+.4f}, with a patient-bootstrap 95% interval from {lower:+.4f} to {upper:+.4f}. Dice improved for {result['patients_improved']} patients and worsened for {result['patients_worsened']}. The bootstrap uses 10,000 resamples with seed 42 and describes variability among these 14 validation patients. Both runs selected checkpoints on this validation set, and this single-seed experiment does not measure training-seed variability.", "", "## Patient comparison", "", "| Patient | Reference Dice | Candidate Dice | Change |", "|---|---:|---:|---:|"]
    for patient, row in paired.sort_values("dice_change").iterrows():
        lines.append(f"| {patient} | {row.reference:.4f} | {row.candidate:.4f} | {row.dice_change:+.4f} |")
    decision = "Use the reference checkpoint for the demo and reported model results. In this run, the sampling candidate has lower mean validation Dice."
    if differences.mean() > 0:
        decision = "The sampling candidate has higher mean validation Dice in this run. Its recall and false-alarm tradeoffs need review and independent confirmation before adoption."
    lines += ["", "## Decision from this experiment", "", decision, "", f"The two largest Dice regressions are {paired.nsmallest(2, 'dice_change').index[0]} and {paired.nsmallest(2, 'dice_change').index[1]}. Patient gains and losses have different magnitudes, so the number of improved patients alone does not determine the mean. The uncertainty interval reflects resampling of these validation patients. Training-seed variability and checkpoint-selection uncertainty require additional experiments."]
    for patient in paired.nsmallest(2, "dice_change").index:
        lines += ["", f"### {patient}", "", f"![Reference worst-error slice]({patient}_reference_worst_slice.png)", "", f"![Candidate worst-error slice]({patient}_candidate_worst_slice.png)"]
    lines += ["", "## Evidence and interpretation", "", "The accompanying reference and candidate PNGs show each model's largest-error slice for every validation patient. Their selected slice can differ between models. Green is correct overlap, red is excess segmentation, and blue is missed annotated tumor. `validation_patient_metrics.csv` contains precision, recall, and full-volume pixel counts. `history.csv` records all epochs, and `sampling.json` records the sampling distribution.", "", "The experiment assesses the proposed sampling strategy using validation patients. A higher validation Dice needs to be considered alongside recall and false alarms. Any selected change still needs confirmation across additional seeds and independent patients, since the original test cases have already informed the diagnosis.", "", f"Reference checkpoint SHA-256 `{result['reference_checkpoint_sha256']}`.", "", f"Candidate checkpoint SHA-256 `{result['candidate_checkpoint_sha256']}`."]
    (candidate.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(), flush=True)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    from src.config import Config
    from src.data import prepare
    from src.train import train
    from src.utils import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    reference = Config(prediction_threshold=0.5)
    candidate = Config(
        output_dir=ROOT / "outputs/experiments/positive_sampling",
        checkpoint_dir=ROOT / "checkpoints/experiments/positive_sampling",
        target_positive_fraction=0.5,
        prediction_threshold=0.5,
    )
    manifest_path = candidate.output_dir / "experiment.json"
    if args.compare_only:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        if (candidate.output_dir / "history.csv").exists():
            raise FileExistsError("An experiment history already exists. Use --compare-only to regenerate the comparison.")
        frame, splits, _ = prepare(reference)
        for filename in ("split.json", "normalization.json"):
            shutil.copy2(reference.output_dir / filename, candidate.output_dir / filename)
        manifest = dict(
            intervention="Expected 50% tumor-containing training slices, replacement sampling",
            epochs=candidate.epochs, seed=candidate.seed, encoder=candidate.encoder,
            batch_size=candidate.batch_size, num_workers=candidate.num_workers,
            target_positive_fraction=candidate.target_positive_fraction,
            reference_checkpoint_sha256=hashlib.sha256((reference.checkpoint_dir / "best.pt").read_bytes()).hexdigest(),
            train_patients=len(splits["train"]), validation_patients=len(splits["val"]),
            training_slices=int(frame.patient.isin(splits["train"]).sum()),
        )
        write_json(manifest_path, manifest)
        start = time.perf_counter()
        train(candidate)
        manifest["training_seconds"] = time.perf_counter() - start
        write_json(manifest_path, manifest)
    compare(reference, candidate, manifest)
    print(f"Experiment complete. Report: {candidate.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
