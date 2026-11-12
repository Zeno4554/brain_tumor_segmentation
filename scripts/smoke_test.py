"""Run an isolated tiny end-to-end check on real or synthetic scans."""
import argparse
import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tifffile
from src.config import Config, ROOT
from src.utils import seed_everything, write_json

def synthetic_data(root: Path) -> None:
    rng = np.random.default_rng(42)
    y, x = np.ogrid[:256, :256]
    brain = ((x - 128) / 95) ** 2 + ((y - 128) / 110) ** 2 < 1
    for p in range(8):
        patient = root / f"synthetic_{p:02d}"
        patient.mkdir(parents=True)
        for index in range(4):
            mask = ((x - rng.integers(95, 160)) ** 2 + (y - rng.integers(95, 160)) ** 2 < rng.integers(8, 20) ** 2) if index in (1, 2) else np.zeros((256, 256), bool)
            image = np.clip(rng.normal(70, 12, (256, 256, 3)), 0, 255) * brain[..., None]
            image[..., 1] += mask * 150
            tifffile.imwrite(patient / f"{patient.name}_{index}.tif", image.astype(np.uint8), photometric="rgb")
            tifffile.imwrite(patient / f"{patient.name}_{index}_mask.tif", mask.astype(np.uint8) * 255)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    from src.baseline import run_baseline
    from src.train import train
    from src.evaluate import evaluate
    from src.data import prepare, MRIDataset
    seed_everything(42)
    with tempfile.TemporaryDirectory(prefix="lgg-smoke-") as temporary:
        source = Config().data_dir
        synthetic = args.synthetic or not any(source.glob("*/*_mask.tif"))
        if synthetic:
            source = Path(temporary) / "kaggle_3m"
            synthetic_data(source)
        output = ROOT / "outputs/smoke"
        checkpoint = ROOT / "checkpoints/smoke"
        # Temporary source paths invalidate previous synthetic-run metadata.
        for filename in ("split.json", "normalization.json"):
            (output / filename).unlink(missing_ok=True)
        config = Config(data_dir=source, output_dir=output, checkpoint_dir=checkpoint, smoke_test=True, epochs=1, encoder_weights=None)
        frame, splits, stats = prepare(config)
        assert len(frame.patient.unique()) == 8
        sample = MRIDataset(frame, stats, augment=True)[0]
        assert sample[0].shape == (3, 256, 256)
        run_baseline(config)
        train(config)
        scores = evaluate(config)
        assert len(scores) == len(splits["test"])
        assert np.isfinite(scores[["unet_dice", "baseline_dice"]]).all().all()
        elapsed = time.perf_counter() - start
        import torch
        if torch.cuda.is_available() and elapsed >= 120:
            raise RuntimeError(f"Smoke pipeline exceeded the 120-second GPU budget ({elapsed:.1f}s)")
        write_json(output / "smoke_result.json", dict(passed=True, seconds=elapsed, synthetic=synthetic, pretrained=False))
        print(f"SMOKE PASSED in {elapsed:.1f}s. Artifacts: {output}")

if __name__ == "__main__":
    main()
