"""Independently audit the prescribed baseline and document publication checks."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audit_baseline() -> None:
    """Reproduce the raw FLAIR rule with Pillow and independent float64 counts."""
    import numpy as np
    import pandas as pd
    from PIL import Image
    from src.config import Config
    from src.data import discover, read_image, read_mask
    from src.baseline import predict_baseline
    from scripts.audit_model import save_case

    config = Config()
    destination = config.output_dir / "publication_checks"
    destination.mkdir(exist_ok=True)
    splits = json.loads((config.output_dir / "split.json").read_text(encoding="utf-8"))
    frame = discover(config)
    saved = pd.read_csv(config.output_dir / "baseline_dice.csv").set_index("patient")
    slices, patients = [], []
    mismatches = 0
    for patient, group in frame[frame.patient.isin(splits["test"])].groupby("patient"):
        tp_total, fp_total, fn_total = 0, 0, 0
        for row in group.sort_values("index").itertuples():
            with Image.open(row.image) as source:
                image = np.array(source)
            with Image.open(row.mask) as source:
                mask = np.array(source) > 0
            np.testing.assert_array_equal(image, read_image(row.image))
            np.testing.assert_array_equal(mask, read_mask(row.mask).astype(bool))
            assert image.shape == (256, 256, 3) and mask.shape == (256, 256)
            flair = image[..., 1].astype(np.float64)
            support = flair[flair > 10]
            cutoff = float(support.mean() + 3 * support.std(ddof=0)) if support.size else None
            pred = flair > cutoff if cutoff is not None else np.zeros(flair.shape, dtype=bool)
            mismatches += int(not np.array_equal(pred, predict_baseline(image.astype(np.float32))))
            tp = int(np.count_nonzero(pred & mask))
            fp = int(np.count_nonzero(pred & ~mask))
            fn = int(np.count_nonzero(~pred & mask))
            tp_total += tp
            fp_total += fp
            fn_total += fn
            slices.append(dict(patient=patient, index=row.index, cutoff=cutoff, support_pixels=int(support.size),
                               maximum_flair=float(flair.max()), cutoff_at_or_above_maximum=cutoff is not None and cutoff >= flair.max(),
                               true_positive_pixels=tp, false_positive_pixels=fp,
                               false_negative_pixels=fn, predicted_pixels=tp + fp,
                               annotated_pixels=tp + fn, image=row.image, mask=row.mask))
        denominator = 2 * tp_total + fp_total + fn_total
        score = 2 * tp_total / (denominator + 1e-7) if denominator else 1.0
        np.testing.assert_allclose(score, saved.loc[patient, "baseline_dice"], atol=1e-7, rtol=0)
        patients.append(dict(patient=patient, independent_volume_dice=score,
                             true_positive_pixels=tp_total, false_positive_pixels=fp_total,
                             false_negative_pixels=fn_total, predicted_pixels=tp_total + fp_total,
                             annotated_pixels=tp_total + fn_total))
    assert mismatches == 0, "Float64/Pillow baseline differs from repository predictions"
    patient_table, slice_table = pd.DataFrame(patients), pd.DataFrame(slices)
    patient_table.to_csv(destination / "baseline_patient_oracle.csv", index=False)
    slice_table.to_csv(destination / "baseline_slice_oracle.csv", index=False)
    findings = dict(
        passed=True, implementation_prediction_mismatch_slices=mismatches,
        patient_count=len(patient_table), slice_count=len(slice_table),
        independent_mean_dice=float(patient_table.independent_volume_dice.mean()),
        independent_median_dice=float(patient_table.independent_volume_dice.median()),
        zero_overlap_patients=int((patient_table.true_positive_pixels == 0).sum()),
        empty_prediction_patients=int((patient_table.predicted_pixels == 0).sum()),
        cutoff_at_or_above_maximum_slices=int(slice_table.cutoff_at_or_above_maximum.sum()),
        empty_support_slices=int((slice_table.support_pixels == 0).sum()),
        pooled_annotated_pixel_recall=float(patient_table.true_positive_pixels.sum() / patient_table.annotated_pixels.sum()),
        channel=1, support_rule="raw FLAIR > 10", cutoff_rule="per-slice mean + 3 * population standard deviation",
        independent_reader="Pillow TIFF", independent_arithmetic="float64, direct TP/FP/FN volume counts",
        published_dice_rule="2*TP/(prediction pixels + annotation pixels + 1e-7), both empty = 1; nonempty zero overlap = 0",
    )
    (destination / "baseline_audit.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    zero_patients = patient_table[patient_table.true_positive_pixels == 0]
    selected = [patient_table.nlargest(1, "independent_volume_dice").iloc[0].patient]
    if len(zero_patients):
        selected.append(zero_patients.nlargest(1, "annotated_pixels").iloc[0].patient)
    for patient in dict.fromkeys(selected):
        row = slice_table[slice_table.patient == patient].nlargest(1, "annotated_pixels").iloc[0]
        image = read_image(row.image)
        truth = read_mask(row['mask']).astype(bool)
        save_case(destination / f"baseline_{patient}.png", image, truth, predict_baseline(image), f"Prescribed raw FLAIR baseline, {patient}, slice {row['index']}, cutoff {row.cutoff:.1f}")
    print(json.dumps(findings, indent=2), flush=True)


def write_report() -> None:
    """Verify checkpoint/run identities and summarize the fixed shipped model."""
    import numpy as np
    import pandas as pd
    import torch

    outputs = ROOT / "outputs"
    destination = outputs / "publication_checks"
    destination.mkdir(exist_ok=True)
    study = outputs / "experiments/normalization_multiseed"
    metrics = json.loads((outputs / "metrics.json").read_text(encoding="utf-8"))
    deployment = json.loads((outputs / "deployment_manifest.json").read_text(encoding="utf-8"))
    calibration = json.loads((ROOT / "checkpoints/inference.json").read_text(encoding="utf-8"))
    baseline = json.loads((destination / "baseline_audit.json").read_text(encoding="utf-8"))
    protocol = json.loads((study / "protocol.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((ROOT / "checkpoints/best.pt").read_bytes()).hexdigest()
    assert digest == metrics["checkpoint_sha256"] == deployment["checkpoint_sha256"] == calibration["checkpoint_sha256"] == protocol["reference_checkpoint_sha256"]
    assert metrics["prediction_threshold"] == deployment["prediction_threshold"] == calibration["prediction_threshold"] == 0.7
    assert metrics["augmentation_seeding"] == "legacy_fixed_transform_seed"
    scores = pd.read_csv(outputs / "per_patient_dice.csv")
    institutions = pd.read_csv(outputs / "per_institution_metrics.csv")
    assert len(scores) == 22 and int(institutions.patients.sum()) == 22
    np.testing.assert_allclose(scores.unet_dice.mean(), metrics["unet_mean"], atol=1e-12)
    for column in ["true_positive_pixels", "false_positive_pixels", "false_negative_pixels", "false_alarms", "tumor_free_slices", "completely_missed_positive_slices", "zero_overlap_positive_slices"]:
        assert int(scores[column].sum()) == metrics[column]
        assert int(institutions[column].sum()) == metrics[column]
    jobs = json.loads((study / "jobs.json").read_text(encoding="utf-8"))
    run_summary = pd.read_csv(study / "run_summary.csv")
    verified = []
    for name, job in jobs.items():
        output = ROOT / job["output_dir"]
        complete = json.loads((output / "completed.json").read_text(encoding="utf-8"))
        history = pd.read_csv(output / "history.csv")
        assert history.epoch.tolist() == list(range(1, 31))
        assert complete["attempt"] == 1
        checkpoint_path = ROOT / job["checkpoint_dir"] / "best.pt"
        assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == complete["checkpoint_sha256"] == job["checkpoint_sha256"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        np.testing.assert_allclose(checkpoint["val_dice"], history.val_dice.max(), atol=1e-12)
        verified.append(dict(run=name, completed_epochs=30, attempt=complete["attempt"], selected_epoch=checkpoint["epoch"], fingerprint_verified=True))
    wide = run_summary[run_summary.threshold == 0.7].pivot(index="seed", columns="mode", values="mean_volume_dice")
    assert (wide.whole_volume > wide.foreground_nonzero).all()
    checks = dict(
        passed=True, shipped_checkpoint_sha256=digest, shipped_epoch=metrics["checkpoint_epoch"],
        shipped_augmentation_seeding=metrics["augmentation_seeding"], shipped_threshold=metrics["prediction_threshold"],
        normalized_study_validation_patients=14, deployed_model_test_patients=22,
        whole_volume_wins_all_seeds=True, completed_runs=verified, baseline_verified=baseline["passed"],
        patient_and_institution_error_totals_match=True,
        controller_recovery="Controller failed during whole_volume_seed43 after a transient status-file rename lock. Training continued through epoch 30. Controller restart verified existing jobs and registered that completed run before starting foreground_nonzero_seed43. All six training jobs completed on attempt 1.",
    )
    demo_path = destination / "demo_verification.json"
    if demo_path.exists():
        demo = json.loads(demo_path.read_text(encoding="utf-8"))
        assert demo["checkpoint_sha256"] == digest and demo["threshold"] == metrics["prediction_threshold"]
        checks["live_demo_identity_and_upload_verified"] = bool(demo["passed"])
    (destination / "verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    lines = ["# Results and deployment verification", "", "## Per-seed validation evidence", "", "These values describe the same 14 validation patients. The 22-patient test result belongs to a different cohort and serves a different purpose. Both normalization conditions use corrected worker seeding. The comparison isolates their normalization settings.", "", "| Seed | Whole-volume Dice | Foreground Dice | Whole minus foreground |", "|---|---:|---:|---:|"]
    for seed, row in wide.iterrows():
        lines.append(f"| {seed} | {row.whole_volume:.4f} | {row.foreground_nonzero:.4f} | {row.whole_volume-row.foreground_nonzero:+.4f} |")
    lines += ["", "Whole-volume normalization scored higher in all three matched runs. The previously reported uncertainty interval includes zero, so the study supports the choice for these runs while leaving population generalization uncertain.", "", "## Shipped checkpoint", "", f"The README, evaluation metrics, and demo identify the original epoch-{metrics['checkpoint_epoch']} checkpoint trained with the older fixed-transform seeding, at inference threshold {metrics['prediction_threshold']:.2f}. Its SHA-256 is `{digest}`. The six corrected-seeding study checkpoints have their own saved identities. The current deployment manifest is `../deployment_manifest.json`.", "", "The original model was evaluated at threshold 0.50 before threshold selection and remeasured at the validation-selected threshold 0.70. This verification adds descriptive errors for that already selected checkpoint and threshold. It provides an audit of the current report, and additional independent patients are needed for confirmatory evaluation.", "", "## Prescribed FLAIR baseline audit", "", f"Pillow TIFF loading, float64 threshold arithmetic, and direct volume-level confusion counts reproduce the repository baseline for all {baseline['patient_count']} patients and {baseline['slice_count']} test slices. Prediction mismatches are {baseline['implementation_prediction_mismatch_slices']}. The independent mean Dice is {baseline['independent_mean_dice']:.4f}, and median Dice is {baseline['independent_median_dice']:.4f}.", "", f"The raw mean-plus-three-standard-deviations cutoff reaches or exceeds the slice maximum on {baseline['cutoff_at_or_above_maximum_slices']} slices. Another {baseline['empty_support_slices']} slices have no FLAIR pixels above 10 and produce empty predictions through the specified fallback. {baseline['zero_overlap_patients']} patients have zero overlapping predicted and annotated pixels. Pooled annotated-pixel recall is {baseline['pooled_annotated_pixel_recall']:.2%}. These measurements show that the prescribed rule is a weak comparator on these scans. The prior approximately 1e-12 values arose from adding epsilon to the numerator when overlap was zero. The shared Dice calculation now returns exact zero for nonempty, zero-overlap volumes and one for two empty volumes; denominator stabilization is retained. This numerical clarification changes the displayed near-zero values, with the reported mean still 0.0117.", "", "The baseline formula remains the original raw channel-1 rule with foreground intensities above 10, computed per slice. A stronger comparator would need a separate validation-based development protocol. The mask annotations represent FLAIR abnormality.", "", "Independent patient and slice counts are in `baseline_patient_oracle.csv` and `baseline_slice_oracle.csv`. Baseline comparison images are saved alongside this report.", "", "## Full errors for the shipped model", "", f"Test mean volume Dice is {metrics['unet_mean']:.4f}, median Dice {metrics['unet_median']:.4f}. Mean patient precision is {metrics['mean_patient_precision']:.2%}, and mean patient recall is {metrics['mean_patient_recall']:.2%}. Pooled pixel precision is {metrics['pooled_precision']:.2%}, and pooled pixel recall is {metrics['pooled_recall']:.2%}. Patient means weight each patient equally; pooled metrics weight their pixel counts.", "", f"There are {metrics['false_positive_pixels']:,} false-positive pixels and {metrics['false_negative_pixels']:,} false-negative pixels across the volumes. Of the false-positive pixels, {metrics['false_positive_pixels_on_tumor_free_slices']:,} occur on tumor-free slices and {metrics['false_positive_pixels_on_positive_slices']:,} on annotated slices. The model predicts no positive pixels on {metrics['completely_missed_positive_slices']} of {metrics['tumor_containing_slices']} annotated slices. Zero true-positive pixels occur on {metrics['zero_overlap_positive_slices']} annotated slices. The zero-overlap definition also detects off-target-only predictions when present. False alarms occur on {metrics['false_alarms']}/{metrics['tumor_free_slices']} tumor-free slices ({metrics['false_alarm_rate']:.2%}). Pixel counts aggregate slices and carry pixel-count units; physical volumes require spacing metadata.", "", "## Institution breakdown", "", "Institution codes come from patient identifiers. These small groups provide descriptive evidence and cannot establish institution-specific causes.", "", "| Institution | Patients | Mean Dice | Mean precision | Mean recall | False alarms | Completely missed annotated slices |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in institutions.itertuples():
        lines.append(f"| {row.institution} | {row.patients} | {row.mean_volume_dice:.4f} | {row.mean_patient_precision:.2%} | {row.mean_patient_recall:.2%} | {row.false_alarms}/{row.tumor_free_slices} | {row.completely_missed_positive_slices}/{row.tumor_containing_slices} |")
    lines += ["", "![Institution means](../per_institution_dice.png)", "", "## Controller recovery and completion", "", checks["controller_recovery"], "", "Every reported study checkpoint has a verified fingerprint, a complete 1-through-30 epoch history, and an attempt-1 completion record. The interrupted component was the progress controller. Training histories, completion records, and selected epochs are preserved in the study directories. Status-file replacement now retries temporary Windows locks. `verification.json` records the checks.", "", "## Files", "", "- `../per_patient_dice.csv` contains complete patient error counts and rates.", "- `../per_institution_metrics.csv` contains institution aggregates.", "- `../metrics.json` contains summary measures and the shipped checkpoint identity.", "- `../deployment_manifest.json` links deployment identity and threshold.", "- `../experiments/normalization_multiseed/report.md` documents all six study runs."]
    lines += ["", "Live demo model-ID, epoch, threshold, and upload verification are recorded in `demo_verification.json`. The study institution aggregates are in `../experiments/normalization_multiseed/per_institution_metrics.csv`, with across-seed summaries in `../experiments/normalization_multiseed/per_institution_seed_summary.csv`."]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("All publication checks passed", flush=True)
    print(json.dumps({key: metrics[key] for key in ["unet_mean", "mean_patient_precision", "mean_patient_recall", "false_positive_pixels", "completely_missed_positive_slices"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only:
        audit_baseline()
    if not args.baseline_only:
        write_report()


if __name__ == "__main__":
    main()
