"""Verify and document an already completed tuned-baseline evaluation."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT
import pandas as pd


def main():
    out = ROOT / 'outputs'
    metrics = json.loads((out / 'metrics.json').read_text())
    result = metrics['baseline_tuned']
    snapshot = json.loads((out / 'baseline_tuned_preservation.json').read_text())
    assert {k: v for k, v in metrics.items() if k != 'baseline_tuned'} == snapshot['existing_metrics']
    for name, expected in {**snapshot['sha256'], **result['protected_artifact_sha256']}.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    frozen = json.loads((out / 'baseline_tuned_config.json').read_text())
    assert hashlib.sha256((out / 'baseline_tuned_config.json').read_bytes()).hexdigest() == result['config_sha256']
    assert hashlib.sha256((out / 'baseline_tuned_search.csv').read_bytes()).hexdigest() == frozen['search_sha256']
    grid = pd.read_csv(out / 'baseline_tuned_search.csv')
    assert len(grid) == 440 and (grid.patients == 88).all()
    assert abs(grid.iloc[0].mean_dice - result['development']['mean_dice']) < 1e-12
    assert result['test']['tumor_free_slices'] == 483 and result['test']['patients'] == 22
    assert len(list((out / 'baseline_tuned_cache/test').glob('*.json'))) == 22
    test = pd.read_csv(out / 'baseline_tuned_dice.csv')
    leading = ['patient', 'baseline_tuned_dice', 'unet_dice']
    if test.columns[:3].tolist() != leading:
        test = test[leading + [c for c in test if c not in leading]]
        test.to_csv(out / 'baseline_tuned_dice.csv', index=False)
    dev = pd.read_csv(out / 'baseline_tuned_development_dice.csv')
    assert len(dev) == 88 and dev.split.value_counts().to_dict() == {'train': 74, 'val': 14}
    assert set(test.patient).isdisjoint(dev.patient)
    p, t, d = result['parameters'], result['test'], result['development']
    readme_path = ROOT / 'README.md'
    readme = readme_path.read_text(encoding='utf-8')
    row = f"| FLAIR threshold (tuned on development patients) | {t['mean_dice']:.4f} | {t['median_dice']:.4f} |"
    if '| FLAIR threshold (tuned on development patients) |' not in readme:
        anchor = '| Prescribed raw FLAIR 3-sigma baseline | 0.0117 | 0.0000 |'
        assert anchor in readme
        readme = readme.replace(anchor, anchor + '\n' + row)
    else:
        assert row in readme
    configuration = (f"Selected tuned-baseline setting: `k={p['k']}`, `E={p['E']}`, "
                     f"`min_area={p['min_area']}`, `largest_3d={p['largest_3d']}`. "
                     f"Development mean/median Dice: {d['mean_dice']:.4f}/{d['median_dice']:.4f}; "
                     f"test false alarms: {t['false_alarms']}/483, pooled precision/recall: "
                     f"{100*t['pooled_precision']:.2f}%/{100*t['pooled_recall']:.2f}%, "
                     f"zero-prediction annotated slices: {t['completely_missed_positive_slices']}/"
                     f"{t['tumor_containing_slices']}, zero-overlap patients: {t['patients_with_zero_overlap']}/22. "
                     'See [the tuned-baseline report](outputs/baseline_tuned_report.md) for development and test errors.')
    if configuration not in readme:
        anchor = '## Threshold selection'
        readme = readme.replace(anchor, configuration + '\n\n' + anchor, 1)
    readme_path.write_text(readme, encoding='utf-8')
    report_configuration = configuration.replace('See [the tuned-baseline report](outputs/baseline_tuned_report.md) for development and test errors.', 'Development and test errors are tabulated below.')
    lines = ['# Tuned classical FLAIR baseline', '', report_configuration, '',
             f"The 440-setting grid was selected only on 74 train + 14 validation patients. Configuration frozen at {frozen['frozen_at_utc']}, before accessing test directories. The fixed comparator was then evaluated once on the 22 test patients; completed patient predictions and measurements are cached.", '',
             frozen['method'] + '.', '', frozen['tie_break'] + '.', '',
             'The selected erosion and minimum area are at the upper bounds of the prescribed grid. This identifies the best tested setting, not a global optimum; the grid was not expanded after seeing test results.', '',
             'This is a descriptive comparison on one split of 22 previously inspected patients. The U-Net margin does not establish a general result. The shipped checkpoint, threshold 0.70, spec baseline and all previous metrics are unchanged.', '',
             '## Development versus test', '', '| Metric | Development (88) | Test (22) |', '|---|---:|---:|']
    for key in ['mean_dice', 'median_dice', 'false_alarms', 'tumor_free_slices', 'false_alarm_rate',
                'pooled_precision', 'pooled_recall', 'true_positive_pixels', 'false_positive_pixels',
                'false_negative_pixels', 'tumor_containing_slices', 'completely_missed_positive_slices',
                'zero_overlap_positive_slices', 'patients_with_zero_overlap']:
        values = [f'{m[key]:.6f}' if isinstance(m[key], float) else str(m[key]) for m in [d, t]]
        lines.append(f'| {key} | {values[0]} | {values[1]} |')
    lines += ['', 'False alarms mean any predicted pixel on a tumor-free slice. A completely missed positive slice has no predicted pixels anywhere; zero-overlap slices may contain off-target predictions. Dice uses the same function as the U-Net: exact zero for nonempty zero-overlap volumes and one for two empty volumes.', '',
              '## Patient-volume comparison', '', '| Patient | Tuned FLAIR Dice | Shipped U-Net Dice (0.70) |', '|---|---:|---:|']
    lines += [f'| {r.patient} | {r.baseline_tuned_dice:.6f} | {r.unet_dice:.6f} |' for r in test.itertuples()]
    lines += ['', '## Figures', '', 'Each figure uses the largest annotated slice in numeric order. Green is truth, blue is tuned FLAIR, and red is the unchanged shipped U-Net at 0.70.', '',
              '![DU_6407](baseline_tuned_DU_6407.png)', '', '![DU_7018](baseline_tuned_DU_7018.png)', '',
              '## Verification and run history', '',
              'The initial serial attempt was stopped during development-only searching to enable three CPU threads. The threaded search restarted from the beginning and completed every patient and every grid setting. No test evaluation occurred during the interrupted attempt.', '',
              'The frozen configuration and grid hashes, development objective, 22 cached test receipts, and exact U-Net file-list correspondence were checked. SHA-256 checks confirm the protected artifacts and source files are unchanged; all pre-existing metrics keys retain their exact values. The isolated synthetic smoke test checks the full grid and independent SciPy morphology/component oracles.', '',
              'Artifacts: `baseline_tuned_search.csv`, `baseline_tuned_config.json`, `baseline_tuned_development_dice.csv`, `baseline_tuned_dice.csv`, `baseline_tuned_test_files.json`, `baseline_tuned_preservation.json`, and the `baseline_tuned` section of `metrics.json`.', '']
    if t['mean_dice'] < 0.1:
        lines += ['The mean Dice is below 0.1. Area/overlap counts for five patients are in `baseline_tuned_low_score_audit.csv`; the independent real-data bug audit must be reviewed before publication.', '']
    (out / 'baseline_tuned_report.md').write_text('\n'.join(lines), encoding='utf-8')
    # Execute only setup and the new results cell. The original baseline cell
    # is retained byte-for-byte as a notebook object and is never executed here.
    import nbformat
    from nbclient import NotebookClient
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    path = ROOT / 'notebooks/02_baseline.ipynb'
    notebook = nbformat.read(path, as_version=4)
    original_cells = copy.deepcopy(notebook.cells[:-2])
    subset = nbformat.v4.new_notebook(cells=[copy.deepcopy(notebook.cells[0]), copy.deepcopy(notebook.cells[-1])], metadata=copy.deepcopy(notebook.metadata))
    for variable, folder in {'IPYTHONDIR': '.ipython', 'JUPYTER_RUNTIME_DIR': '.jupyter_runtime'}.items():
        os.environ.setdefault(variable, str(out / folder))
    NotebookClient(subset, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    notebook.cells[-1] = subset.cells[-1]
    assert notebook.cells[:-2] == original_cells
    nbformat.write(notebook, path)
    print('Verified preserved artifacts, results, grid, file lists, documentation, and new notebook cell.')
    print(json.dumps(dict(parameters=p, development=d, test=t), indent=2))


if __name__ == '__main__':
    main()
