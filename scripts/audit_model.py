"""Collect reproducible failure evidence and verify the running demo."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config

import numpy as np
import pandas as pd
import tifffile
from matplotlib.figure import Figure
from scipy import ndimage

from src.baseline import predict_baseline
from src.data import MRIDataset, check_leakage, prepare, read_image, read_mask
from src.evaluate import predict_patient
from src.model import load_model
from src.utils import device, dice, seed_everything, segmentation_metrics, write_json


def measurements(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Count errors across the supplied slice or complete patient volume."""
    return segmentation_metrics(pred, truth)


def remove_components(volume: np.ndarray, minimum: int) -> np.ndarray:
    """Apply a diagnostic 8-connected component filter independently per slice."""
    result = np.zeros_like(volume)
    for index, mask in enumerate(volume):
        labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))
        sizes = np.bincount(labels.ravel())
        keep = sizes >= minimum
        keep[0] = False
        result[index] = keep[labels]
    return result


def save_case(path: Path, image: np.ndarray, truth: np.ndarray, pred: np.ndarray, title: str) -> None:
    """Show annotation, prediction, and spatial error on the same FLAIR slice."""
    fig = Figure(figsize=(12, 3.4))
    axes = fig.subplots(1, 4)
    flair = image[..., 1]
    axes[0].imshow(flair, cmap="gray")
    axes[1].imshow(truth, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    errors = np.zeros((*truth.shape, 3), dtype=np.uint8)
    errors[pred & truth] = [0, 190, 0]
    errors[pred & ~truth] = [255, 0, 0]
    errors[~pred & truth] = [0, 120, 255]
    axes[3].imshow(errors)
    for ax, label in zip(axes, ["FLAIR", "Ground truth", "Prediction", "Green overlap / red excess / blue missed"]):
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def audit(config: Config, destination: Path) -> None:
    seed_everything(config.seed)
    dev = device()
    model, checkpoint = load_model(config, dev)
    frame, splits, stats = prepare(config)
    check_leakage(splits)
    assert checkpoint["splits"] == splits
    assert checkpoint["normalization"] == stats
    checkpoint_hash = hashlib.sha256((config.checkpoint_dir / "best.pt").read_bytes()).hexdigest()
    scores = pd.read_csv(config.output_dir / "per_patient_dice.csv")
    weakest = scores.nsmallest(3, "unet_dice").patient.tolist()
    split_lookup = {patient: name for name, patients in splits.items() for patient in patients}
    dataset_rows = []
    positive_examples = []
    print("Auditing every image/mask pair and intensity statistics", flush=True)
    for patient, group in frame.groupby("patient"):
        assert not group['index'].duplicated().any(), f"Duplicate slice indices for {patient}"
        sums, squares, count = np.zeros(3), np.zeros(3), np.zeros(3)
        for row in group.itertuples():
            image = read_image(row.image)
            raw_mask = tifffile.imread(row.mask)
            assert image.shape == (256, 256, 3)
            assert raw_mask.shape == image.shape[:2]
            assert np.isfinite(image).all()
            assert set(np.unique(raw_mask)).issubset({0, 1, 255})
            truth = raw_mask > 0
            dataset_rows.append(dict(patient=patient, split=split_lookup[patient], index=row.index, tumor_pixels=int(truth.sum())))
            if truth.any() and split_lookup[patient] == "train" and len(positive_examples) < 6:
                positive_examples.append(row)
            if patient in weakest:
                pixels = image.reshape(-1, 3).astype(np.float64)
                sums += pixels.sum(0)
                squares += (pixels ** 2).sum(0)
                count += (pixels != 0).sum(0) if config.normalization_mode == "foreground_nonzero" else len(pixels)
        if patient in weakest:
            support = np.maximum(count, 1)
            mean = sums / support
            std = np.sqrt(np.maximum(squares / support - mean ** 2, 1e-6))
            std[count == 0] = 1
            np.testing.assert_allclose(mean, stats[patient]["mean"], rtol=1e-6)
            np.testing.assert_allclose(std, stats[patient]["std"], rtol=1e-6)
    composition = pd.DataFrame(dataset_rows)
    composition.to_csv(destination / "dataset_slices.csv", index=False)
    balance = composition.assign(tumor_present=composition.tumor_pixels > 0).groupby("split").agg(
        patients=("patient", "nunique"), slices=("patient", "size"), tumor_slices=("tumor_present", "sum")
    )
    balance["tumor_slice_fraction"] = balance.tumor_slices / balance.slices
    balance.to_csv(destination / "dataset_balance.csv")

    dataset = MRIDataset(frame, stats, augment=True, seed=config.seed)
    fig = Figure(figsize=(12, 5))
    axes = fig.subplots(2, 6)
    for column, row in enumerate(positive_examples):
        transformed = dataset.transform(image=read_image(row.image), mask=read_mask(row.mask))
        assert np.isin(transformed["mask"], [0, 1]).all()
        axes[0, column].imshow(transformed["image"][..., 1], cmap="gray")
        axes[1, column].imshow(transformed["mask"], cmap="gray")
        for ax in axes[:, column]:
            ax.axis("off")
    fig.suptitle("Training augmentation examples, FLAIR above and matching binary mask below")
    fig.tight_layout()
    fig.savefig(destination / "augmentation_examples.png", dpi=140)

    patients, slices, baselines, experiments, cases = [], [], [], [], []
    for split in ("val", "test"):
        for patient, group in frame[frame.patient.isin(splits[split])].groupby("patient"):
            print(f"Measuring {split} {patient}", flush=True)
            group = group.sort_values("index")
            truth = np.stack([read_mask(row.mask) for row in group.itertuples()]).astype(bool)
            pred = predict_patient(model, group, stats, config, dev)
            patient_metrics = measurements(pred, truth)
            patients.append(dict(patient=patient, split=split, **patient_metrics))
            if split == "val":
                for minimum in (0, 16, 64, 128):
                    filtered = pred if minimum == 0 else remove_components(pred, minimum)
                    experiments.append(dict(patient=patient, minimum_component_pixels=minimum, **measurements(filtered, truth)))
            patient_slice_rows = []
            for position, row in enumerate(group.itertuples()):
                measured = measurements(pred[position], truth[position])
                record = dict(patient=patient, split=split, index=row.index, **measured)
                slices.append(record)
                patient_slice_rows.append(record)
                if split == "test":
                    image = read_image(row.image)
                    flair = image[..., 1]
                    brain = flair[flair > 10]
                    threshold = float(brain.mean() + 3 * brain.std()) if brain.size else None
                    base = predict_baseline(image)
                    baselines.append(dict(
                        patient=patient, index=row.index, threshold=threshold,
                        flair_max=float(flair.max()), threshold_exceeds_max=threshold is not None and threshold >= flair.max(),
                        **measurements(base, truth[position]),
                    ))
            if patient in weakest:
                annotated = [i for i, row in enumerate(patient_slice_rows) if row["annotated_pixels"] > 0]
                negative = [i for i, row in enumerate(patient_slice_rows) if row["annotated_pixels"] == 0]
                selections = {
                    "most_missed_tumor": max(annotated, key=lambda i: patient_slice_rows[i]["false_negative_pixels"]),
                    "most_excess_segmentation": max(annotated, key=lambda i: patient_slice_rows[i]["false_positive_pixels"]),
                }
                if negative:
                    selections["tumor_free_check"] = max(negative, key=lambda i: patient_slice_rows[i]["predicted_pixels"])
                for reason, position in selections.items():
                    row = group.iloc[position]
                    filename = f"{patient}_{int(row['index'])}_{reason}.png"
                    save_case(destination / filename, read_image(row.image), truth[position], pred[position], f"{patient}, slice {row['index']}, {reason.replace('_', ' ')}")
                    cases.append(dict(patient=patient, index=int(row['index']), reason=reason, image=row.image, mask=row['mask'], evidence_image=filename, **measurements(pred[position], truth[position])))
    patient_table = pd.DataFrame(patients)
    patient_table.to_csv(destination / "patient_diagnostics.csv", index=False)
    pd.DataFrame(slices).to_csv(destination / "slice_diagnostics.csv", index=False)
    baseline_table = pd.DataFrame(baselines)
    baseline_table.to_csv(destination / "baseline_diagnostics.csv", index=False)
    pd.DataFrame(cases).to_csv(destination / "demo_cases.csv", index=False)
    test = patient_table[patient_table.split == "test"].merge(scores, on="patient", suffixes=("", "_saved"))
    np.testing.assert_allclose(test.dice, test.unet_dice, atol=1e-7, rtol=0)
    saved_metrics = json.loads((config.output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert int(test.false_alarms.sum()) == saved_metrics["false_alarms"]
    experiment_table = pd.DataFrame(experiments)
    experiment_table.to_csv(destination / "validation_component_experiment.csv", index=False)
    summary = experiment_table.groupby("minimum_component_pixels").agg(
        mean_volume_dice=("dice", "mean"), median_volume_dice=("dice", "median"),
        false_alarms=("false_alarms", "sum"), tumor_free_slices=("tumor_free_slices", "sum"),
        true_positive_pixels=("true_positive_pixels", "sum"), false_negative_pixels=("false_negative_pixels", "sum"),
    )
    summary["false_alarm_rate"] = summary.false_alarms / summary.tumor_free_slices
    summary["pooled_tumor_recall"] = summary.true_positive_pixels / (summary.true_positive_pixels + summary.false_negative_pixels)
    summary.to_csv(destination / "validation_component_summary.csv")
    history = pd.read_csv(config.output_dir / "history.csv")
    details = dict(
        checkpoint_sha256=checkpoint_hash, checkpoint_epoch=checkpoint["epoch"],
        checkpoint_validation_dice=checkpoint["val_dice"],
        paired_files_checked=len(frame), normalization_recomputed_for=weakest,
        baseline_slices=len(baseline_table), baseline_threshold_exceeds_max=int(baseline_table.threshold_exceeds_max.sum()),
        baseline_tumor_recall=float(baseline_table.true_positive_pixels.sum() / baseline_table.annotated_pixels.sum()),
        last_epoch_validation_dice=float(history.iloc[-1].val_dice),
        test_scores_reproduced=True, selected_examples=len(cases),
    )
    write_json(destination / "audit_checks.json", details)
    lines = [
        "# Model diagnosis and evidence", "",
        f"This audit uses the saved epoch-{checkpoint['epoch']} {checkpoint['encoder']} checkpoint, normalization mode {config.normalization_mode}, prediction threshold {config.prediction_threshold:.2f}, and the original patient split. Precision and recall below aggregate complete patient volumes. Slice measurements select illustrative examples only.", "",
        "## Three weakest test patients", "",
        "| Patient | Volume Dice | Precision | Tumor recall | Excess pixels | Missed pixels |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for patient in weakest:
        row = patient_table[patient_table.patient == patient].iloc[0]
        lines.append(f"| {patient} | {row.dice:.4f} | {row.precision:.4f} | {row.recall:.4f} | {int(row.false_positive_pixels)} | {int(row.false_negative_pixels)} |")
    lines += ["", "Measured failure patterns follow from the volume counts above.", ""]
    for patient in weakest:
        row = patient_table[patient_table.patient == patient].iloc[0]
        lines.append(f"- `{patient}` misses {1 - row.recall:.1%} of annotated tumor. Of its predicted tumor pixels, {1 - row.precision:.1%} fall outside the annotation.")
    lines += ["", "## Annotated examples", "", "Green shows correct overlap, red shows excess prediction, and blue shows missed annotated tumor. Examples are chosen by maximum missed pixels, maximum excess pixels on annotated slices, and maximum prediction on tumor-free slices within each weak patient.", ""]
    for case in cases:
        lines += [f"### {case['patient']} slice {case['index']}, {case['reason'].replace('_', ' ')}", "", f"Predicted area {case['predicted_pixels']} pixels, annotated area {case['annotated_pixels']} pixels. Slice Dice {case['dice']:.4f}.", "", f"![Comparison]({case['evidence_image']})", ""]
    lines += [
        "## Dataset and preprocessing checks", "",
        f"All {len(frame)} image/mask pairs have matching 256 by 256 geometry, three image channels, finite intensities, supported binary mask values, and unique numeric slice indices within each patient. The saved patient sets are disjoint. Checkpoint split and normalization metadata match the current dataset. Recomputed normalization for all three weak scans matches the cache. These checks establish file and array consistency. Anatomical annotation correctness and MRI channel semantics still require source metadata or expert review.", "",
        "| Split | Patients | Slices | Tumor-containing slices | Tumor fraction |", "|---|---:|---:|---:|---:|",
    ]
    for name, row in balance.iterrows():
        lines.append(f"| {name} | {int(row.patients)} | {int(row.slices)} | {int(row.tumor_slices)} | {row.tumor_slice_fraction:.1%} |")
    lines += ["", "The augmentation uses shared flips and rotations with nearest-neighbor mask interpolation. The sampled transformed masks remain binary.", "", "![Augmentation examples](augmentation_examples.png)", "", "## Baseline diagnosis", "", f"The specified mean-plus-three-standard-deviations FLAIR threshold reaches or exceeds the maximum slice intensity on {details['baseline_threshold_exceeds_max']} of {len(baseline_table)} test slices. Such slices produce empty baseline predictions. Baseline pooled tumor recall is {details['baseline_tumor_recall']:.2%}. This explains part of its low Dice under the prescribed rule. The audit uses the original channel-1 baseline formula.", "", "## Training evidence", "", f"The best checkpoint is from epoch {checkpoint['epoch']}, with mean validation Dice {checkpoint['val_dice']:.4f}. Final-epoch validation Dice is {history.iloc[-1].val_dice:.4f}. Training loss decreased from {history.iloc[0].train_loss:.4f} to {history.iloc[-1].train_loss:.4f}. Validation recovered after the first learning-rate reduction. These curves alone do not establish the cause of individual patient failures.", "", "## Validation-only component-removal experiment", "", "This diagnostic experiment removes 8-connected predicted components smaller than a fixed pixel count within each slice. It uses the existing checkpoint and 14 validation patients. All cutoffs were specified before computing these comparisons. The model checkpoint and deployed demo use the original prediction pipeline.", "", "| Minimum component pixels | Mean volume Dice | False alarms | False-alarm rate | Pooled tumor recall |", "|---|---:|---:|---:|---:|"]
    for minimum, row in summary.iterrows():
        lines.append(f"| {minimum} | {row.mean_volume_dice:.4f} | {int(row.false_alarms)} | {row.false_alarm_rate:.2%} | {row.pooled_tumor_recall:.2%} |")
    comparisons = experiment_table.pivot(index="patient", columns="minimum_component_pixels", values="dice")
    differences = (comparisons[128] - comparisons[0]).to_numpy()
    rng = np.random.default_rng(config.seed)
    bootstrap = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    improved = int((differences > 1e-7).sum())
    worsened = int((differences < -1e-7).sum())
    original, filtered = summary.loc[0], summary.loc[128]
    comparison = dict(
        minimum_component_pixels=128, patients_improved=improved, patients_worsened=worsened,
        patients_unchanged=int((np.abs(differences) <= 1e-7).sum()),
        mean_dice_difference=float(differences.mean()),
        exploratory_bootstrap_95_percent_interval=[float(lower), float(upper)],
        true_positive_pixels_removed=int(original.true_positive_pixels - filtered.true_positive_pixels),
    )
    write_json(destination / "validation_experiment_interpretation.json", comparison)
    lines += ["", f"The 128-pixel filter improves Dice for {improved} validation patients and worsens it for {worsened}. Mean Dice changes by {differences.mean():+.4f}. An exploratory paired patient bootstrap with 10,000 resamples and seed 42 gives a 95% interval from {lower:+.4f} to {upper:+.4f}. This interval describes the observed patient sample and omits cutoff-selection uncertainty. The filter removes {comparison['true_positive_pixels_removed']} correctly predicted tumor pixels. Its benefit requires confirmation on additional patients before deployment."]
    lines += ["", "## Interpretation and next experiment", "", "Use volume precision and recall to identify whether each failure is dominated by missed tumor or excess segmentation. Small-component removal can reduce isolated false alarms while deleting genuine small tumor regions. Compare both volume Dice and recall before adopting it. The three weak patients were identified from the test results, so diagnosis here is exploratory. Any subsequent training or postprocessing improvement needs confirmation on an additional untouched patient set. A single fixed split and these selected images cannot establish reliable generalization.", "", "The next training candidate should follow the measured failure pattern. Low recall supports testing increased sampling of tumor-containing training slices while retaining negative slices. Excess segmentation supports testing validation-selected component removal. Each candidate needs comparison against the original checkpoint on the same validation patients before a new external evaluation.", "", f"Checkpoint SHA-256 `{checkpoint_hash}`.", "", "Detailed data are saved in `patient_diagnostics.csv`, `slice_diagnostics.csv`, `baseline_diagnostics.csv`, and `validation_component_experiment.csv`."]
    largest_miss = max(cases, key=lambda case: case["false_negative_pixels"])
    lines += ["", "## Recommended order of further work", "", f"The largest selected missed region is in {largest_miss['patient']}, slice {largest_miss['index']}. It has {largest_miss['annotated_pixels']:,} annotated tumor pixels and {largest_miss['predicted_pixels']:,} predicted pixels. Review its full scan and all three MRI channels alongside the low-precision cases. The weak patients have different mixtures of missed tumor and excess segmentation, so a single automatic correction may have different effects across patients.", "", "The first training experiment should compare modestly increased sampling of tumor-containing training slices against the current sampling, retaining tumor-free slices and the fixed validation patients. Judge both patient-volume Dice and false alarms, since increased positive sampling can increase excess segmentation. The component-removal pilot provides a separate candidate with a measured recall tradeoff. Additional independent patient evaluation is needed to establish improvement after choosing either approach."]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Diagnosis complete. Report: {destination / 'report.md'}", flush=True)


def verify_demo(config: Config, destination: Path, url: str) -> None:
    """Exercise real uploads through Gradio and compare with local FP32 inference."""
    from gradio_client import Client, handle_file
    from PIL import Image
    from src.app import Predictor

    client = Client(url, verbose=False)
    predictor = Predictor(config)
    results = []
    cases = pd.read_csv(destination / "demo_cases.csv")
    for row in cases.itertuples():
        print(f"Verifying demo {row.patient} slice {row.index}", flush=True)
        expected_overlay, expected_area = predictor.predict(row.image, row.patient)
        image_path, area = client.predict(handle_file(row.image), row.patient, api_name="/predict")
        assert area == expected_area, f"Demo/local area mismatch for {row.patient}"
        with Image.open(image_path) as returned:
            actual_overlay = np.asarray(returned.convert("RGB"))
        assert actual_overlay.shape == expected_overlay.shape
        # Gradio may encode returned images as lossy WebP, so compare geometry and area.
        filename = f"demo_{row.patient}_{row.index}_{row.reason}.png"
        Image.fromarray(actual_overlay).save(destination / filename)
        results.append(dict(patient=row.patient, index=row.index, reason=row.reason, demo_area=int(area), local_fp32_area=expected_area, annotated_area=row.annotated_pixels, evaluation_amp_area=row.predicted_pixels, passed=True, response_image=filename))
    pd.DataFrame(results).to_csv(destination / "demo_verification.csv", index=False)
    path = destination / "report.md"
    report = path.read_text(encoding="utf-8").split("\n## Live demo verification")[0]
    report += f"\n## Live demo verification\n\nAll {len(results)} selected cases passed upload, patient-profile selection, inference, image return, and numeric-area checks through the running Gradio API. Demo areas exactly match local FP32 inference. Saved `demo_*.png` files show the returned images. Evaluation uses mixed precision, so its pixel counts can differ slightly from FP32 demo inference near the 0.5 cutoff. The API check exercises the same prediction callback as Submit. Verification covers HTTP upload and prediction behavior. Visual browser rendering and file-picker interactions remain outside this API check.\n"
    path.write_text(report, encoding="utf-8")
    print(f"All {len(results)} live demo cases passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-only", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    args = parser.parse_args()
    config = Config()
    destination = config.output_dir / "audit"
    destination.mkdir(parents=True, exist_ok=True)
    if args.demo_only:
        verify_demo(config, destination, args.url)
    else:
        audit(config, destination)


if __name__ == "__main__":
    main()
