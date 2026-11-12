"""Run and monitor six matched normalization trials, with restartable job records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "outputs/experiments/normalization_multiseed"
CHECKPOINTS = ROOT / "checkpoints/experiments/normalization_multiseed"
SEEDS = (42, 43, 44)
MODES = ("whole_volume", "foreground_nonzero")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "outputs/huggingface_cache"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "outputs/torch_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))


def save_status(path: Path, value: dict) -> None:
    """Atomically publish a progress record while monitoring tools read it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    for attempt in range(60):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 59:
                raise
            time.sleep(0.25)


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_study() -> None:
    """Cache both scan normalizations and record image-derived diagnostics."""
    import numpy as np
    import pandas as pd
    import torch
    from matplotlib.figure import Figure
    from src.config import Config
    from src.data import normalize, prepare, read_image, read_mask

    original = Config()
    frame, splits, whole = prepare(original)
    for mode in MODES:
        directory = STUDY / "cache" / mode
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original.output_dir / "split.json", directory / "split.json")
        if mode == "whole_volume":
            shutil.copy2(original.output_dir / "normalization.json", directory / "normalization.json")
    foreground_config = Config(
        output_dir=STUDY / "cache/foreground_nonzero",
        checkpoint_dir=CHECKPOINTS / "cache", normalization_mode="foreground_nonzero",
    )
    _, foreground_splits, foreground = prepare(foreground_config)
    assert foreground_splits == splits
    lookup = {patient: name for name, patients in splits.items() for patient in patients}
    records, examples, composition = [], [], []
    for patient, group in frame.groupby("patient"):
        pre_equals, post_equals = True, True
        for row in group.itertuples():
            if pre_equals or post_equals:
                image = read_image(row.image)
                pre_equals = pre_equals and np.array_equal(image[..., 0], image[..., 1])
                post_equals = post_equals and np.array_equal(image[..., 2], image[..., 1])
            if lookup[patient] != "test":
                area = int(read_mask(row.mask).sum())
                composition.append(dict(patient=patient, split=lookup[patient], index=row.index, tumor_pixels=area))
                if len(examples) < 3 and area > 100 and patient not in {e.patient for e in examples}:
                    examples.append(row)
        record = dict(patient=patient, split=lookup[patient], site=patient.split("_")[1], slices=len(group), pre_identical_to_flair=pre_equals, post_identical_to_flair=post_equals)
        for channel in range(3):
            count = foreground[patient]["count"][channel]
            record.update({
                f"foreground_fraction_ch{channel}": count / (len(group) * 256 * 256),
                f"whole_mean_ch{channel}": whole[patient]["mean"][channel],
                f"whole_std_ch{channel}": whole[patient]["std"][channel],
                f"foreground_mean_ch{channel}": foreground[patient]["mean"][channel],
                f"foreground_std_ch{channel}": foreground[patient]["std"][channel],
            })
        records.append(record)
    pd.DataFrame(records).to_csv(STUDY / "scan_diagnostics.csv", index=False)
    pd.DataFrame(composition).to_csv(STUDY / "development_slice_composition.csv", index=False)
    fig = Figure(figsize=(10, 9))
    axes = fig.subplots(3, 3)
    for row_axes, example in zip(axes, examples):
        image = read_image(example.image)
        images = [image[..., 1], normalize(image, whole[example.patient])[..., 1], normalize(image, foreground[example.patient])[..., 1]]
        for ax, content, title in zip(row_axes, images, ["Original FLAIR", "Whole-volume normalization", "Nonzero-foreground normalization"]):
            ax.imshow(content, cmap="gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(STUDY / "normalization_examples.png", dpi=130)
    manifest = dict(
        conditions=list(MODES), seeds=list(SEEDS), epochs=30, batch_size=8,
        encoder="resnet34", encoder_weights="imagenet", training_workers=2,
        persistent_training_workers=True, validation_workers=0,
        augmentation_seeding="torch_initial_seed_per_worker_with_continuing_epoch_streams",
        sampling="natural shuffled slices, no replacement",
        training_checkpoint_threshold=0.5, primary_comparison_threshold=0.7,
        secondary_comparison_threshold=0.5,
        foreground_definition="Each channel independently excludes exactly zero-valued pixels from scan-level mean/std; normalized zero-valued pixels stay zero. This is an image-derived foreground, not anatomical brain extraction.",
        training_patients=splits["train"], validation_patients=splits["val"],
        test_labels_or_predictions_used=0,
        reference_checkpoint_sha256=fingerprint(ROOT / "checkpoints/best.pt"),
        reference_inference_sha256=fingerprint(ROOT / "checkpoints/inference.json"),
        reference_metrics=json.loads((ROOT / "outputs/metrics.json").read_text(encoding="utf-8")),
        torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        source_sha256={name: fingerprint(ROOT / name) for name in ["src/data.py", "src/train.py", "src/model.py", "src/config.py"]},
        interpretation_rule="Compare matched seeds and patient-level Dice, recall, and false alarms. Recommend foreground normalization only with consistent improvements and acceptable error tradeoffs; provide uncertainty and preserve the deployed reference during the study.",
    )
    save_status(STUDY / "protocol.json", manifest)
    print("Study preprocessing and protocol saved", flush=True)


def train_job(mode: str, seed: int, attempt: int) -> None:
    from src.config import Config
    from src.train import train

    name = f"{mode}_seed{seed}"
    suffix = "" if attempt == 1 else f"_attempt{attempt}"
    output = STUDY / f"{name}{suffix}"
    checkpoint = CHECKPOINTS / f"{name}{suffix}"
    output.mkdir(parents=True, exist_ok=True)
    for filename in ("split.json", "normalization.json"):
        shutil.copy2(STUDY / "cache" / mode / filename, output / filename)
    config = Config(
        output_dir=output, checkpoint_dir=checkpoint, seed=seed,
        normalization_mode=mode, prediction_threshold=0.7,
    )
    start = time.perf_counter()
    path = train(config)
    save_status(output / "completed.json", dict(
        mode=mode, seed=seed, attempt=attempt, checkpoint=str(path.relative_to(ROOT)),
        checkpoint_sha256=fingerprint(path), training_seconds=time.perf_counter() - start,
    ))


def summarize_study() -> None:
    """Evaluate only validation scans and compare matched seed outcomes."""
    import numpy as np
    import pandas as pd
    from matplotlib.figure import Figure
    from src.config import Config
    from src.data import prepare, read_image, read_mask
    from src.evaluate import predict_probabilities
    from src.model import load_model
    from src.utils import device, seed_everything
    from scripts.audit_model import measurements, save_case

    manifest = json.loads((STUDY / "protocol.json").read_text(encoding="utf-8"))
    completed = json.loads((STUDY / "jobs.json").read_text(encoding="utf-8"))
    seed_everything(42)
    dev = device()
    records = []
    for name, entry in completed.items():
        output = ROOT / entry["output_dir"]
        config = Config(output_dir=output, checkpoint_dir=ROOT / entry["checkpoint_dir"], prediction_threshold=0.7, normalization_mode=entry["mode"])
        model, checkpoint = load_model(config, dev)
        frame, splits, stats = prepare(config)
        assert checkpoint["splits"] == splits
        assert checkpoint["normalization"] == stats
        for patient, group in frame[frame.patient.isin(splits["val"])].groupby("patient"):
            group = group.sort_values("index")
            truth = np.stack([read_mask(row.mask) for row in group.itertuples()]).astype(bool)
            probabilities = predict_probabilities(model, group, stats, config, dev)
            for threshold in (0.5, 0.7):
                pred = probabilities >= threshold
                measured = measurements(pred, truth)
                records.append(dict(run=name, mode=entry["mode"], seed=entry["seed"], patient=patient, site=patient.split("_")[1], threshold=threshold, **measured))
        print(f"Evaluated {name} on validation patients", flush=True)
        del model
    table = pd.DataFrame(records)
    table.to_csv(STUDY / "validation_patient_metrics.csv", index=False)
    run_summary = table.groupby(["mode", "seed", "threshold"]).agg(
        mean_volume_dice=("dice", "mean"), median_volume_dice=("dice", "median"),
        mean_patient_recall=("recall", "mean"), mean_patient_precision=("precision", "mean"),
        false_alarms=("false_alarms", "sum"), tumor_free_slices=("tumor_free_slices", "sum"),
        false_positive_pixels=("false_positive_pixels", "sum"), false_negative_pixels=("false_negative_pixels", "sum"),
        completely_missed_positive_slices=("completely_missed_positive_slices", "sum"),
    ).reset_index()
    run_summary["false_alarm_rate"] = run_summary.false_alarms / run_summary.tumor_free_slices
    run_summary.to_csv(STUDY / "run_summary.csv", index=False)
    primary = table[table.threshold == 0.7]
    paired = primary.pivot(index=["seed", "patient"], columns="mode", values="dice")
    paired["dice_change"] = paired.foreground_nonzero - paired.whole_volume
    paired.to_csv(STUDY / "paired_patient_dice.csv")
    matrix = paired.dice_change.unstack("patient").loc[list(SEEDS)].to_numpy()
    rng = np.random.default_rng(42)
    bootstrap = []
    for _ in range(10000):
        seeds = rng.integers(0, matrix.shape[0], matrix.shape[0])
        patients = rng.integers(0, matrix.shape[1], matrix.shape[1])
        bootstrap.append(float(matrix[np.ix_(seeds, patients)].mean()))
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    aggregate = run_summary[run_summary.threshold == 0.7].groupby("mode").agg(
        mean_dice_across_seeds=("mean_volume_dice", "mean"), std_dice_across_seeds=("mean_volume_dice", "std"),
        mean_recall_across_seeds=("mean_patient_recall", "mean"), mean_precision_across_seeds=("mean_patient_precision", "mean"),
        mean_false_alarm_rate=("false_alarm_rate", "mean"),
        mean_missed_positive_slices=("completely_missed_positive_slices", "mean"),
    )
    aggregate.to_csv(STUDY / "condition_summary.csv")
    seed_paired = run_summary[run_summary.threshold == 0.7].pivot(index="seed", columns="mode", values="mean_volume_dice")
    seed_delta = seed_paired.foreground_nonzero - seed_paired.whole_volume
    candidate, reference = aggregate.loc["foreground_nonzero"], aggregate.loc["whole_volume"]
    consistent_gain = bool((seed_delta > 0).all() and candidate.mean_recall_across_seeds >= reference.mean_recall_across_seeds and candidate.mean_false_alarm_rate <= reference.mean_false_alarm_rate)
    result = dict(
        mean_paired_dice_change=float(matrix.mean()),
        matched_seed_dice_changes={str(seed): float(value) for seed, value in seed_delta.items()},
        exploratory_crossed_seed_patient_bootstrap_95_percent_interval=[float(lower), float(upper)],
        patients_improved_averaging_seeds=int((matrix.mean(axis=0) > 1e-7).sum()),
        patients_worsened_averaging_seeds=int((matrix.mean(axis=0) < -1e-7).sum()),
        consistent_dice_recall_false_alarm_gain=consistent_gain,
        test_patients_evaluated=0, validation_patients=matrix.shape[1], seeds=list(SEEDS),
        primary_threshold=0.7,
    )
    save_status(STUDY / "comparison.json", result)
    assert fingerprint(ROOT / "checkpoints/best.pt") == manifest["reference_checkpoint_sha256"]
    assert fingerprint(ROOT / "checkpoints/inference.json") == manifest["reference_inference_sha256"]
    fig = Figure(figsize=(10, 4))
    axes = fig.subplots(1, 2)
    for mode in MODES:
        selected = run_summary[(run_summary.threshold == 0.7) & (run_summary['mode'] == mode)]
        axes[0].plot(selected.seed, selected.mean_volume_dice, marker="o", label=mode)
        axes[1].plot(selected.seed, selected.mean_patient_recall, marker="o", label=mode)
    axes[0].set_title("Mean validation patient-volume Dice")
    axes[1].set_title("Mean validation patient tumor recall")
    for ax in axes:
        ax.set_xticks(list(SEEDS))
        ax.set_xlabel("Training seed")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(STUDY / "matched_seed_results.png", dpi=150)
    # Select the lowest and highest changes averaged over seeds, then compare the
    # same slice using the corresponding seed-42 models for spatial evidence.
    patient_delta = paired.dice_change.groupby("patient").mean().sort_values()
    selected_patients = list(dict.fromkeys([patient_delta.index[0], patient_delta.index[-1]]))
    models = {}
    configurations = {}
    normalizations = {}
    for mode in MODES:
        entry = completed[f"{mode}_seed42"]
        config = Config(output_dir=ROOT / entry["output_dir"], checkpoint_dir=ROOT / entry["checkpoint_dir"], normalization_mode=mode, prediction_threshold=0.7)
        models[mode], _ = load_model(config, dev)
        frame, splits, normalizations[mode] = prepare(config)
        configurations[mode] = config
    for patient in selected_patients:
        group = frame[frame.patient == patient].sort_values("index")
        truth = np.stack([read_mask(row.mask) for row in group.itertuples()]).astype(bool)
        predictions = {mode: predict_probabilities(models[mode], group, normalizations[mode], configurations[mode], dev) >= 0.7 for mode in MODES}
        errors = np.logical_xor(predictions["foreground_nonzero"], truth).sum(axis=(1, 2))
        position = int(np.argmax(errors))
        row = group.iloc[position]
        for mode in MODES:
            save_case(STUDY / f"{patient}_{mode}_comparison.png", read_image(row.image), truth[position], predictions[mode][position], f"{patient}, slice {row['index']}, seed 42, {mode}")
    lines = ["# Worker seeding and normalization experiment", "", "Six matched 30-epoch runs compare whole-volume normalization with per-channel, per-patient nonzero-foreground normalization. Both conditions use corrected worker-specific augmentation seeds and persistent training workers. Exactly zero-valued pixels are excluded from foreground statistics and remain zero after normalization. The foreground is image-derived and includes non-brain head tissues; it is not a brain-extraction mask.", "", "Seeds are 42, 43, and 44. Each run uses ImageNet-initialized ResNet34, batch size 8, natural slice sampling, the original Dice+BCE loss and optimizer/scheduler settings, 74 training patients, and the same 14 validation patients. Training checkpoint selection uses volume Dice at threshold 0.5. The prespecified primary comparison uses threshold 0.7 for both conditions. Secondary threshold-0.5 results are in `run_summary.csv`.", "", "## Matched-seed validation results", "", "| Method | Seed | Mean volume Dice | Median volume Dice | Mean patient recall | False alarms | Completely missed positive slices |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in run_summary[run_summary.threshold == 0.7].itertuples():
        lines.append(f"| {row.mode} | {row.seed} | {row.mean_volume_dice:.4f} | {row.median_volume_dice:.4f} | {row.mean_patient_recall:.2%} | {int(row.false_alarms)}/{int(row.tumor_free_slices)} | {int(row.completely_missed_positive_slices)} |")
    lines += ["", "## Across-seed comparison", "", f"Foreground normalization changes mean Dice by {matrix.mean():+.4f}, averaged across matched seeds and patients. The exploratory crossed seed/patient bootstrap 95% interval is [{lower:+.4f}, {upper:+.4f}]. It resamples seed pairs and patient identities jointly across both methods, using 10,000 replicates and seed 42. The interval has limited precision with three seeds and omits checkpoint-selection uncertainty. Validation patients have been reused during earlier development, so this study cannot establish independent generalization.", "", f"Average Dice improves for {result['patients_improved_averaging_seeds']} validation patients and worsens for {result['patients_worsened_averaging_seeds']}.", "", "![Matched-seed results](matched_seed_results.png)", "", "## Recommendation", ""]
    if consistent_gain:
        lines += ["Foreground normalization shows higher mean Dice in every matched seed, with average recall at least as high and average false-alarm rate no higher. It is a promising candidate for additional patient-group validation and independent-data confirmation."]
    else:
        lines += ["These results do not demonstrate consistent improvement across Dice, recall, and false alarms. Use the established deployed reference while interpreting the measured tradeoffs. The corrected worker-seeding implementation is available for subsequent training, and each new checkpoint records its normalization and seeding settings."]
    lines += ["", "## Same-slice spatial comparisons", "", "These examples use the same slice for both seed-42 models. Patients have the lowest and highest average Dice changes across seeds. Green indicates overlap, red excess prediction, and blue missed annotation."]
    for patient in selected_patients:
        lines += ["", f"### {patient}", "", f"![Whole-volume]({patient}_whole_volume_comparison.png)", "", f"![Foreground]({patient}_foreground_nonzero_comparison.png)"]
    lines += ["", "## Preprocessing evidence and limitations", "", "`preprocessing_checks.json` records actual two-worker, two-epoch reproducibility checks, worker/epoch stream diversity, binary-mask geometry checks, background handling, blank-channel support, cache invalidation, and legacy normalization equivalence. `scan_diagnostics.csv` contains institution codes, duplicated-channel flags, foreground fractions, and both sets of scan statistics. Duplicated channels may reflect documented replacement of missing sequences with FLAIR. Source documentation defines the target as FLAIR abnormality. Annotation consistency and anatomical correctness require source-data or expert review.", "", "![Normalization examples](normalization_examples.png)", "", "Per-patient measurements, completely missed positive-slice counts, and false-positive/negative pixel counts are in `validation_patient_metrics.csv`. The test set contributes zero label or prediction measurements to this experiment. Its unlabeled scan intensities appear in cached per-patient statistics, as in the existing pipeline. The original test cases were inspected previously, so an additional independent dataset is needed for final confirmation.", "", "## Runtime and reproducibility", "", "`protocol.json` records software, source fingerprints, original checkpoint identity, fixed settings, and selection rules. `jobs.json` identifies all six completed checkpoints. Each job has `history.csv`, `completed.json`, normalization metadata, and a log. Running this script again verifies completed job identities and continues the unfinished queue. Interrupted jobs start fresh in a separate attempt directory, retaining their partial artifacts."]
    training_hours = sum(entry["training_seconds"] for entry in completed.values()) / 3600
    lines += ["", f"The six runs used {training_hours:.2f} hours of cumulative training time."]
    (STUDY / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(aggregate.to_string(), flush=True)
    print(json.dumps(result, indent=2), flush=True)


def launch(command: list[str], log_path: Path) -> int:
    """Run one stage with its combined output saved under the study directory."""
    with log_path.open("a", encoding="utf-8") as log:
        return subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


def run_study(compare_only: bool) -> None:
    STUDY.mkdir(parents=True, exist_ok=True)
    if not (STUDY / "preprocessing_checks.json").exists():
        if launch([sys.executable, "-u", "scripts/verify_preprocessing.py"], STUDY / "preprocessing.log"):
            raise RuntimeError("Preprocessing verification failed. See preprocessing.log")
    if not compare_only and not (STUDY / "protocol.json").exists():
        if launch([sys.executable, "-u", __file__, "--prepare"], STUDY / "preparation.log"):
            raise RuntimeError("Study preparation failed. See preparation.log")
    jobs_path = STUDY / "jobs.json"
    jobs = json.loads(jobs_path.read_text(encoding="utf-8")) if jobs_path.exists() else {}
    if not compare_only:
        for seed in SEEDS:
            for mode in MODES:
                name = f"{mode}_seed{seed}"
                if name in jobs:
                    checkpoint = ROOT / jobs[name]["checkpoint_dir"] / "best.pt"
                    assert fingerprint(checkpoint) == jobs[name]["checkpoint_sha256"]
                    print(f"Verified completed {name}", flush=True)
                    continue
                attempts = [1, 2]
                for attempt in attempts:
                    suffix = "" if attempt == 1 else f"_attempt{attempt}"
                    output = STUDY / f"{name}{suffix}"
                    checkpoint_dir = CHECKPOINTS / f"{name}{suffix}"
                    output.mkdir(parents=True, exist_ok=True)
                    complete_path = output / "completed.json"
                    if complete_path.exists():
                        completed = json.loads(complete_path.read_text(encoding="utf-8"))
                        assert fingerprint(ROOT / completed["checkpoint"]) == completed["checkpoint_sha256"]
                        success = True
                    elif (output / "history.csv").exists():
                        print(f"Preserving partial {name} attempt {attempt}", flush=True)
                        success = False
                    else:
                        print(f"Starting {name}, attempt {attempt}", flush=True)
                        start = time.perf_counter()
                        with (output / "training.log").open("w", encoding="utf-8") as log:
                            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                            process = subprocess.Popen([sys.executable, "-u", __file__, "--worker", "--mode", mode, "--seed", str(seed), "--attempt", str(attempt)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
                            while process.poll() is None:
                                latest = None
                                history_path = output / "history.csv"
                                if history_path.exists():
                                    with history_path.open(newline="", encoding="utf-8") as history:
                                        rows = list(csv.DictReader(history))
                                        if rows:
                                            latest = rows[-1]
                                try:
                                    save_status(STUDY / "status.json", dict(
                                        status="training", current_job=name, attempt=attempt,
                                        completed_jobs=len(jobs), total_jobs=6, latest_epoch=latest,
                                        elapsed_job_seconds=time.perf_counter() - start,
                                    ))
                                except PermissionError:
                                    print("Progress file is temporarily locked; training continues", flush=True)
                                time.sleep(20)
                        success = process.returncode == 0 and complete_path.exists()
                        if success:
                            completed = json.loads(complete_path.read_text(encoding="utf-8"))
                    if success:
                        jobs[name] = dict(mode=mode, seed=seed, output_dir=str(output.relative_to(ROOT)), checkpoint_dir=str(checkpoint_dir.relative_to(ROOT)), checkpoint_sha256=completed["checkpoint_sha256"], training_seconds=completed["training_seconds"])
                        save_status(jobs_path, jobs)
                        print(f"Completed {name}", flush=True)
                        break
                else:
                    save_status(STUDY / "status.json", dict(status="failed", current_job=name, completed_jobs=len(jobs)))
                    raise RuntimeError(f"Both attempts failed for {name}. See the job training logs.")
    if len(jobs) != 6:
        raise RuntimeError("All six matched jobs are required before comparison")
    save_status(STUDY / "status.json", dict(status="comparing", completed_jobs=6, total_jobs=6))
    if launch([sys.executable, "-u", __file__, "--summarize"], STUDY / "comparison.log"):
        raise RuntimeError("Validation comparison failed. See comparison.log")
    save_status(STUDY / "status.json", dict(status="complete", completed_jobs=6, total_jobs=6, report=str((STUDY / "report.md").relative_to(ROOT))))
    print(f"STUDY COMPLETE: {STUDY / 'report.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.prepare:
        prepare_study()
    elif args.worker:
        train_job(args.mode, args.seed, args.attempt)
    elif args.summarize:
        summarize_study()
    else:
        run_study(args.compare_only)


if __name__ == "__main__":
    main()
