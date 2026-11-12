"""Development-only classical FLAIR search, followed by one frozen test evaluation.

2D components use 8-connectivity; 3D components use 26-connectivity.
Erosion uses a Euclidean disk of radius E; population statistics use float64.
No normalization, fitting of weights, or neural inference enters this baseline.
"""
import hashlib
import itertools
import json
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .config import Config
from .data import check_leakage, read_image, read_mask
from .utils import dice, segmentation_metrics, write_json

GRID = dict(k=[0.5 + 0.25 * i for i in range(11)], E=[0, 3, 6, 9],
            min_area=[0, 20, 50, 100, 200], largest_3d=[False, True])
S2 = np.ones((3, 3), bool)
S3 = np.ones((3, 3, 3), bool)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patient_files(config, patients):
    """Same mask-led pairing and numeric ordering as src.data.discover.

    Only enumerate requested patient directories: test files are not opened or
    enumerated during the development search.
    """
    rows = []
    for patient in sorted(patients):
        masks = list((config.data_dir / patient).glob('*_mask.tif'))
        assert masks, f'No slices for {patient}'
        for mask in masks:
            image = Path(str(mask).replace('_mask', ''))
            assert image.is_file()
            rows.append(dict(patient=patient, index=int(mask.stem.rsplit('_', 2)[1]),
                             image=str(image), mask=str(mask)))
    frame = pd.DataFrame(rows).sort_values(['patient', 'index']).reset_index(drop=True)
    assert not frame.duplicated(['patient', 'index']).any()
    return frame


def load_volume(group):
    group = group.sort_values('index')
    flair = np.stack([read_image(r.image)[..., 1] for r in group.itertuples()])
    truth = np.stack([read_mask(r.mask) for r in group.itertuples()]).astype(bool)
    assert flair.shape == truth.shape and np.isfinite(flair).all()
    return flair, truth


def brain_masks(flair, E):
    masks = []
    for image in flair:
        support = ndi.binary_fill_holes(image > 10)
        labels, count = ndi.label(support, S2)
        if count:
            sizes = np.bincount(labels.ravel()); sizes[0] = 0
            support = labels == sizes.argmax()
        if E:
            # Padding makes image-edge behavior identical to disk erosion with
            # a zero-valued exterior, including masks that touch the image edge.
            distance = ndi.distance_transform_edt(np.pad(support, 1))[1:-1, 1:-1]
            support = distance > E
        masks.append(support)
    return np.stack(masks)


def candidates(flair, brain, k):
    values = flair[brain].astype(np.float64)
    if not values.size:
        return np.zeros(flair.shape, bool)
    return brain & (flair > values.mean() + k * values.std())


def slice_labels(pred):
    labels, sizes = [], []
    for image in pred:
        lab, _ = ndi.label(image, S2)
        labels.append(lab)
        sizes.append(np.bincount(lab.ravel()))
    return labels, sizes


def remove_small(labels, sizes, min_area):
    result = []
    for lab, counts in zip(labels, sizes):
        keep = counts >= min_area
        keep[0] = False
        result.append(keep[lab])
    return np.stack(result)


def largest_component(pred):
    positions = np.nonzero(pred)
    if not len(positions[0]):
        return pred.copy()
    box = tuple(slice(int(p.min()), int(p.max()) + 1) for p in positions)
    labels, _ = ndi.label(pred[box], S3)
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    result = np.zeros_like(pred)
    result[box] = labels == sizes.argmax()
    return result


def predict(flair, parameters):
    brain = brain_masks(flair, parameters['E'])
    initial = candidates(flair, brain, parameters['k'])
    pred = remove_small(*slice_labels(initial), parameters['min_area'])
    return largest_component(pred) if parameters['largest_3d'] else pred


def patient_search(group, grid):
    flair, truth = load_volume(group)
    result = {}
    for E in grid['E']:
        brain = brain_masks(flair, E)
        for k in grid['k']:
            labels, sizes = slice_labels(candidates(flair, brain, k))
            for area in grid['min_area']:
                pred = remove_small(labels, sizes, area)
                for largest in grid['largest_3d']:
                    measured = largest_component(pred) if largest else pred
                    result[(k, E, area, largest)] = dice(measured, truth)
    return result


def search(config, frame, grid=None):
    grid = grid or GRID
    settings = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    scores = {tuple(s.values()): [] for s in settings}
    groups = list(frame.groupby('patient'))
    # SciPy releases the GIL. Threads share imports and keep memory bounded;
    # map preserves patient order for deterministic score aggregation.
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = pool.map(lambda pair: patient_search(pair[1], grid), groups)
        for number, ((patient, _), result) in enumerate(zip(groups, results), 1):
            for key, value in result.items():
                scores[key].append(value)
            print(f'Development search {number}/{len(groups)}: {patient}', flush=True)
    rows = [dict(**s, mean_dice=float(np.mean(scores[tuple(s.values())])),
                 median_dice=float(np.median(scores[tuple(s.values())])),
                 patients=frame.patient.nunique()) for s in settings]
    table = pd.DataFrame(rows).sort_values(
        ['mean_dice', 'E', 'min_area', 'largest_3d', 'k'],
        ascending=[False, True, True, True, True]).reset_index(drop=True)
    table.to_csv(config.output_dir / 'baseline_tuned_search.csv', index=False)
    best = table.iloc[0]
    return dict(k=float(best.k), E=int(best.E), min_area=int(best.min_area),
                largest_3d=bool(best.largest_3d)), float(best.mean_dice)


def summarize(table):
    tp, fp, fn = (int(table[c].sum()) for c in
                  ['true_positive_pixels', 'false_positive_pixels', 'false_negative_pixels'])
    counts = ['false_alarms', 'tumor_free_slices', 'tumor_containing_slices',
              'completely_missed_positive_slices', 'zero_overlap_positive_slices',
              'false_positive_pixels_on_tumor_free_slices',
              'false_positive_pixels_on_positive_slices']
    result = dict(patients=len(table), mean_dice=float(table.baseline_tuned_dice.mean()),
                  median_dice=float(table.baseline_tuned_dice.median()),
                  pooled_precision=tp / (tp + fp) if tp + fp else None,
                  pooled_recall=tp / (tp + fn) if tp + fn else None,
                  true_positive_pixels=tp, false_positive_pixels=fp, false_negative_pixels=fn,
                  patients_with_zero_overlap=int((table.true_positive_pixels == 0).sum()),
                  **{c: int(table[c].sum()) for c in counts})
    result['false_alarm_rate'] = result['false_alarms'] / result['tumor_free_slices'] if result['tumor_free_slices'] else None
    return result


def evaluate_fixed(config, frame, parameters, name):
    """Persist each completed patient; resuming uses cached measurements/masks."""
    cache = config.output_dir / 'baseline_tuned_cache' / name
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for patient, group in frame.groupby('patient'):
        receipt = cache / f'{patient}.json'
        if receipt.exists():
            record = json.loads(receipt.read_text())
            assert record['parameters'] == parameters
            measured = record['metrics']
        else:
            flair, truth = load_volume(group)
            pred = predict(flair, parameters)
            measured = segmentation_metrics(pred, truth)
            np.savez_compressed(cache / f'{patient}.npz', pred=pred)
            write_json(receipt, dict(parameters=parameters, metrics=measured))
        rows.append(dict(patient=patient, **measured))
    return pd.DataFrame(rows).rename(columns={'dice': 'baseline_tuned_dice'})


def comparison_figures(config, test, reference):
    """U-Net inference is used only for the requested frozen-model outlines."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from .evaluate import predict_patient
    from .model import load_model
    from .utils import device
    dev = device()
    model, checkpoint = load_model(config, dev)
    assert config.prediction_threshold == 0.70
    assert checkpoint['splits'] == json.loads((config.output_dir / 'split.json').read_text())
    assert checkpoint['normalization'] == json.loads((config.output_dir / 'normalization.json').read_text())
    for short in ['DU_6407', 'DU_7018']:
        group = test[test.patient.str.contains(short)].sort_values('index')
        assert group.patient.nunique() == 1
        patient = group.patient.iloc[0]
        flair, truth = load_volume(group)
        position = int(truth.sum(axis=(1, 2)).argmax())
        pred = np.load(config.output_dir / 'baseline_tuned_cache/test' / f'{patient}.npz')['pred']
        neural = predict_patient(model, group.iloc[[position]], checkpoint['normalization'], config, dev)[0]
        fig = Figure(figsize=(8, 8)); ax = fig.subplots()
        ax.imshow(flair[position], cmap='gray')
        for mask, color in [(truth[position], 'lime'), (pred[position], 'blue'), (neural, 'red')]:
            if mask.any():
                ax.contour(mask, levels=[0.5], colors=[color], linewidths=1.1)
        ax.legend(handles=[Line2D([0], [0], color=c, label=t) for c, t in
                  [('lime', 'Truth'), ('blue', 'Tuned FLAIR'), ('red', 'U-Net (0.70)')]], loc='lower right')
        ax.set_title(f'{patient}, slice {int(group.iloc[position]["index"])}\nLargest annotated slice; volume scores in baseline_tuned_dice.csv')
        ax.axis('off'); fig.tight_layout()
        fig.savefig(config.output_dir / f'baseline_tuned_{short}.png', dpi=160)


def run_tuned_baseline(config=None):
    config = config or Config()
    out = config.output_dir
    metrics_path = out / 'metrics.json'
    original = json.loads(metrics_path.read_text())
    if 'baseline_tuned' in original:
        print('Using completed frozen baseline results; no test reevaluation.')
        return pd.read_csv(out / 'baseline_tuned_dice.csv')
    protected = [p for p in out.iterdir() if p.is_file() and p.suffix in {'.csv', '.json', '.png'}
                 and not p.name.startswith('baseline_tuned') and p.name != 'metrics.json']
    protected += [config.checkpoint_dir / name for name in ['best.pt', 'inference.json']]
    hashes = {str(p): fingerprint(p) for p in protected}
    splits = json.loads((out / 'split.json').read_text()); check_leakage(splits)
    assert (len(splits['train']), len(splits['val']), len(splits['test'])) == (74, 14, 22)
    development = patient_files(config, splits['train'] + splits['val'])
    frozen_path = out / 'baseline_tuned_config.json'
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        assert frozen['grid'] == GRID and frozen['development_patients'] == sorted(splits['train'] + splits['val'])
        assert frozen['search_sha256'] == fingerprint(out / 'baseline_tuned_search.csv')
        parameters = frozen['parameters']
    else:
        parameters, score = search(config, development)
        frozen = dict(parameters=parameters, grid=GRID, development_patients=sorted(splits['train'] + splits['val']),
                      development_objective=score, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                      search_sha256=fingerprint(out / 'baseline_tuned_search.csv'),
                      method='raw FLAIR index 1; filled >10 mask; largest 8-connected slice component; Euclidean disk erosion; patient-pooled float64 population mean/std; slice 8-connectivity; volume 26-connectivity',
                      tie_break='exact mean Dice ties: smaller E, smaller min_area, no 3D filter, then smaller k')
        write_json(frozen_path, frozen)
    frozen_hash = fingerprint(frozen_path)
    print('FROZEN before any test files are opened:', parameters, flush=True)
    dev_table = evaluate_fixed(config, development, parameters, 'development')
    dev_table['split'] = dev_table.patient.map(lambda p: 'train' if p in splits['train'] else 'val')
    dev_table.to_csv(out / 'baseline_tuned_development_dice.csv', index=False)
    # These are the first test-directory accesses. Assert exact lists against
    # archived U-Net slice evaluation AND normalization image-file signatures.
    test = patient_files(config, splits['test'])
    reference_slices = pd.read_csv(out / 'per_slice_errors.csv').sort_values(['patient', 'index'])
    assert list(test[['patient', 'index']].itertuples(index=False, name=None)) == list(reference_slices[['patient', 'index']].itertuples(index=False, name=None))
    normalization = json.loads((out / 'normalization.json').read_text())
    for patient, group in test.groupby('patient'):
        expected = normalization[patient]['signature']
        actual = [[r.image, Path(r.image).stat().st_size, Path(r.image).stat().st_mtime_ns] for r in group.itertuples()]
        assert actual == expected, f'U-Net file-list mismatch: {patient}'
        assert all(r.mask == r.image.replace('.tif', '_mask.tif') for r in group.itertuples())
    write_json(out / 'baseline_tuned_test_files.json', dict(config_sha256=frozen_hash, files=test.to_dict('records'), identical_to_unet=True))
    test_table = evaluate_fixed(config, test, parameters, 'test')
    reference = pd.read_csv(out / 'per_patient_dice.csv')[['patient', 'unet_dice']]
    assert set(reference.patient) == set(test_table.patient)
    test_table = test_table.merge(reference, on='patient', validate='one_to_one')
    leading = ['patient', 'baseline_tuned_dice', 'unet_dice']
    test_table = test_table[leading + [c for c in test_table if c not in leading]]
    test_table.to_csv(out / 'baseline_tuned_dice.csv', index=False)
    test_metrics = summarize(test_table)
    if test_metrics['mean_dice'] < 0.1:
        audit = test_table.head(5)[['patient', 'predicted_pixels', 'annotated_pixels', 'true_positive_pixels']]
        print('Low-score area/overlap audit:\n' + audit.to_string(index=False), flush=True)
        audit.to_csv(out / 'baseline_tuned_low_score_audit.csv', index=False)
    comparison_figures(config, test, reference)
    assert fingerprint(frozen_path) == frozen_hash
    assert all(fingerprint(Path(p)) == h for p, h in hashes.items()), 'Protected artifact changed'
    assert json.loads(metrics_path.read_text()) == original
    section = dict(parameters=parameters, config_sha256=frozen_hash,
                   development=summarize(dev_table), test=test_metrics,
                   test_evaluations=1, test_file_lists_identical_to_unet=True,
                   scope='Descriptive comparison on one split of 22 previously inspected patients; parameters selected only on 88 development patients.',
                   completely_missed_slice_definition='annotated slice with no predicted positive pixels anywhere',
                   protected_artifact_sha256=hashes)
    write_json(metrics_path, {**original, 'baseline_tuned': section})
    assert {k: v for k, v in json.loads(metrics_path.read_text()).items() if k != 'baseline_tuned'} == original
    print(json.dumps(section, indent=2))
    return test_table


if __name__ == '__main__':
    run_tuned_baseline()
