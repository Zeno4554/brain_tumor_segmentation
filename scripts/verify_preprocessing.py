"""Check worker randomness, joint geometry, and scan normalization before training."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker_sequence(seed: int, frame, stats: dict) -> list[list[str]]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from src.data import MRIDataset, seed_worker

    loader = DataLoader(
        MRIDataset(frame, stats, augment=True, seed=seed),
        batch_size=1, num_workers=2, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed), persistent_workers=True,
    )
    sequences = []
    for _ in range(2):
        signatures = []
        for image, mask in loader:
            assert np.isin(mask.numpy(), [0, 1]).all()
            signatures.append(hashlib.sha256(image.numpy().tobytes() + mask.numpy().tobytes()).hexdigest())
        sequences.append(signatures)
    return sequences


def main() -> None:
    import numpy as np
    import pandas as pd
    import tifffile
    from src.config import Config
    from src.data import MRIDataset, discover, normalize, patient_statistics, read_image
    from src.utils import write_json

    config = Config()
    frame = discover(config)
    stats = json.loads((config.output_dir / "normalization.json").read_text(encoding="utf-8"))
    split = json.loads((config.output_dir / "split.json").read_text(encoding="utf-8"))
    training = frame[frame.patient.isin(split["train"])]
    repeated = pd.concat([training.iloc[[len(training) // 2]]] * 24, ignore_index=True)
    first = worker_sequence(42, repeated, stats)
    second = worker_sequence(42, repeated, stats)
    other = worker_sequence(43, repeated, stats)
    assert first == second, "Repeated run with the same seed changed augmentation results"
    assert first != other, "Different seeds produced the same augmentation stream"
    assert first[0] != first[1], "Augmentation stream repeated between epochs"
    assert first[0][::2] != first[0][1::2], "Workers produced identical augmentation sequences"

    transform = MRIDataset(repeated, stats, augment=True).transform
    y, x = np.ogrid[:256, :256]
    mask = (((x - 91) / 30) ** 2 + ((y - 147) / 42) ** 2 < 1).astype(np.float32)
    image = np.repeat(mask[..., None], 3, axis=-1)
    disagreements = []
    for _ in range(24):
        transformed = transform(image=image, mask=mask)
        assert np.isin(transformed["mask"], [0, 1]).all()
        disagreement = float(np.mean((transformed["image"][..., 0] >= 0.5) != transformed["mask"]))
        disagreements.append(disagreement)
        assert disagreement < 0.002, "Image and mask geometry diverged"

    legacy_row = training.iloc[0]
    legacy_image = read_image(legacy_row.image)
    legacy_stats = stats[legacy_row.patient]
    expected = (legacy_image - np.asarray(legacy_stats["mean"], np.float32)) / np.asarray(legacy_stats["std"], np.float32)
    np.testing.assert_array_equal(normalize(legacy_image, legacy_stats), expected)

    with tempfile.TemporaryDirectory(prefix="lgg-normalization-check-") as directory:
        root = Path(directory)
        images = [np.array([[[0, 0, 0], [10, 4, 0]], [[20, 8, 0], [0, 0, 0]]], dtype=np.float32)]
        images.append(images[0] * 2)
        rows = []
        for index, sample in enumerate(images):
            path = root / f"scan_{index}.tif"
            tifffile.imwrite(path, sample, photometric="rgb")
            rows.append(dict(patient="scan", image=str(path)))
        temporary = Config(data_dir=root, output_dir=root / "outputs", checkpoint_dir=root / "checkpoints", normalization_mode="foreground_nonzero")
        sample_frame = pd.DataFrame(rows)
        foreground = patient_statistics(temporary, sample_frame)["scan"]
        values = np.stack(images)
        np.testing.assert_allclose(foreground["mean"], [22.5, 9, 0])
        assert foreground["count"] == [4, 4, 0]
        assert foreground["std"][2] == 1
        normalized = np.stack([normalize(sample, foreground) for sample in images])
        assert np.all(normalized[values == 0] == 0)
        for channel in (0, 1):
            selected = normalized[..., channel][values[..., channel] != 0]
            np.testing.assert_allclose(selected.mean(), 0, atol=1e-6)
            np.testing.assert_allclose(selected.std(), 1, atol=1e-6)
        assert patient_statistics(temporary, sample_frame)["scan"] == foreground
        temporary.normalization_mode = "whole_volume"
        whole = patient_statistics(temporary, sample_frame)["scan"]
        np.testing.assert_allclose(whole["mean"], [11.25, 4.5, 0])
        assert whole.get("mode", "whole_volume") == "whole_volume"

    destination = config.output_dir / "experiments/normalization_multiseed"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "preprocessing_checks.json", dict(
        passed=True, worker_count=2, samples_per_epoch=24, epochs_checked=2,
        repeated_seed_reproducible=True, worker_streams_differ=True,
        epoch_streams_differ=True, seed_change_changes_stream=True,
        masks_binary=True, max_geometry_disagreement_fraction=max(disagreements),
        legacy_normalization_unchanged=True, foreground_scan_statistics_correct=True,
        background_stays_zero=True, empty_channel_supported=True,
        cache_mode_invalidation_verified=True,
    ))
    print("PREPROCESSING CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
