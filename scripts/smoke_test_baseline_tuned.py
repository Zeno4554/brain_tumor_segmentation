"""Isolated synthetic classical-baseline smoke check; no shipped outputs touched."""
import argparse
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import ndimage as ndi
from scripts.smoke_test import synthetic_data
from src.config import Config
from src.baseline_tuned import (GRID, brain_masks, candidates, evaluate_fixed,
                                largest_component, patient_files, predict, search, summarize)
from src.utils import dice, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', action='store_true', required=True)
    parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='tuned-flair-smoke-') as folder:
        root = Path(folder)
        synthetic_data(root / 'data')
        for patient in (root / 'data').iterdir():
            for suffix in ['.tif', '_mask.tif']:
                (patient / f'{patient.name}_3{suffix}').rename(patient / f'{patient.name}_10{suffix}')
        config = Config(data_dir=root / 'data', output_dir=root / 'outputs', checkpoint_dir=root / 'checkpoints')
        dev = patient_files(config, [f'synthetic_{i:02d}' for i in range(6)])
        assert dev[dev.patient == 'synthetic_00']['index'].tolist() == [0, 1, 2, 10]
        parameters, objective = search(config, dev)
        assert len(__import__('pandas').read_csv(config.output_dir / 'baseline_tuned_search.csv')) == 440
        write_json(config.output_dir / 'baseline_tuned_config.json', parameters)
        test = patient_files(config, ['synthetic_06', 'synthetic_07'])
        result = evaluate_fixed(config, test, parameters, 'test')
        assert objective > 0.8 and summarize(result)['mean_dice'] > 0.8
        cached = evaluate_fixed(config, test, parameters, 'test')
        assert result.equals(cached)
        # Independent direct SciPy oracle for morphology/postprocessing.
        rng = np.random.default_rng(7)
        volume = rng.uniform(20, 100, (5, 32, 32)).astype(np.float32)
        volume[:, :2] = 0
        for E in GRID['E']:
            support = brain_masks(volume, E)
            y, x = np.ogrid[-E:E+1, -E:E+1]
            disk = x*x + y*y <= E*E
            base = brain_masks(volume, 0)
            oracle_support = np.stack([ndi.binary_erosion(s, structure=disk) for s in base]) if E else base
            assert np.array_equal(support, oracle_support)
            for area in [0, 20]:
                p = dict(k=0.75, E=E, min_area=area, largest_3d=True)
                raw = candidates(volume, support, p['k'])
                cleaned = np.zeros_like(raw)
                for i, image in enumerate(raw):
                    labels, count = ndi.label(image, np.ones((3, 3)))
                    for label in range(1, count+1):
                        component = labels == label
                        if component.sum() >= area:
                            cleaned[i] |= component
                labels, count = ndi.label(cleaned, np.ones((3, 3, 3)))
                expected = np.zeros_like(cleaned)
                if count:
                    selected = max(range(1, count+1), key=lambda label: (labels == label).sum())
                    expected = labels == selected
                assert np.array_equal(predict(volume, p), expected)
        assert dice(np.ones((1, 2, 2), bool), np.zeros((1, 2, 2), bool)) == 0
        assert not predict(np.zeros((2, 8, 8)), parameters).any()
        # 3D filter keeps a component across adjacent numeric slices.
        blob = np.zeros((3, 8, 8), bool); blob[:, 2:4, 2:4] = True; blob[0, 7, 7] = True
        assert largest_component(blob).sum() == 12
        print('SYNTHETIC TUNED BASELINE SMOKE PASSED', parameters, summarize(result))


if __name__ == '__main__':
    main()
