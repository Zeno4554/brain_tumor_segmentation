"""Export a small, portable evidence snapshot without changing original runs."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs'
DESTINATION = ROOT / 'reports'
ROOT_JSON = {'metrics.json', 'deployment_manifest.json', 'model_evaluation_summary.json',
             'split.json', 'baseline_tuned_config.json', 'baseline_tuned_verification.json'}
PLOTS = {'patient_dice.png', 'per_institution_dice.png', 'training_curves.png',
         'threshold_curves.png', 'matched_seed_results.png'}
EXCLUDED = {'smoke', 'audit', 'huggingface_cache', 'torch_cache', '.ipython', '.matplotlib',
            '.jupyter_runtime', 'baseline_tuned_cache', 'notebook_archive'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected(path):
    relative = path.relative_to(SOURCE)
    if any(part in EXCLUDED for part in relative.parts):
        return False
    if path.suffix in {'.md', '.csv'}:
        return True
    if path.name in PLOTS:
        return True
    if path.suffix == '.json':
        if len(relative.parts) == 1:
            return path.name in ROOT_JSON
        return path.name in {'comparison.json', 'selection.json', 'completed.json',
                             'condition_summary.json', 'integrity_checks.json',
                             'verification.json', 'shipped_candidate_check.json',
                             'recall_optimization_verification.json'}
    return False


def portable(value):
    if isinstance(value, dict):
        return {portable(k): portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str):
        text = value.replace(str(ROOT).replace('\\', '\\\\'), '.')
        text = text.replace(str(ROOT), '.').replace(str(ROOT).replace('\\', '/'), '.')
        return text.replace('\\', '/')
    return value


def main():
    files = [p for p in SOURCE.rglob('*') if p.is_file() and selected(p)]
    mapping = {p.resolve(): DESTINATION / p.relative_to(SOURCE) for p in files}
    previous_path = DESTINATION/'manifest.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else []
    targets = {p.resolve() for p in mapping.values()}
    for entry in previous:
        stale = (ROOT/entry['destination']).resolve()
        assert stale.is_relative_to(DESTINATION.resolve())
        if stale not in targets and stale.exists():
            if sha(stale) != entry['exported_sha256']:
                raise ValueError(f'Preserving edited snapshot file: {stale}')
            stale.unlink()
    manifest = []
    for path in sorted(files):
        target = mapping[path.resolve()]
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.json':
            value = portable(json.loads(path.read_text(encoding='utf-8')))
            data = (json.dumps(value, indent=2) + '\n').encode('utf-8')
        elif path.suffix == '.md':
            text = path.read_text(encoding='utf-8')
            # Do not embed MRI examples in the public evidence snapshot.
            def image_link(match):
                location = (path.parent / match.group(2)).resolve()
                if location not in mapping:
                    return '*MRI illustration retained in the local generated report.*'
                return match.group(0)
            text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', image_link, text)
            text = portable(text)
            data = text.encode('utf-8')
        elif path.suffix == '.csv':
            data = portable(path.read_text(encoding='utf-8')).encode('utf-8')
        else:
            data = path.read_bytes()
        target.write_bytes(data)
        manifest.append(dict(source=path.relative_to(ROOT).as_posix(),
                             destination=target.relative_to(ROOT).as_posix(),
                             source_sha256=sha(path), exported_sha256=sha(target), bytes=len(data)))
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    (DESTINATION/'README.md').write_text(
        '# Verified results snapshot\n\n'
        'These files summarize completed local runs; exporting them does not rerun the test set. '
        'Original measurements remain in ignored `outputs/`. Machine-specific paths in copied JSON '
        'are made relative. MRI illustrations, raw data, probability arrays and weights stay local.\n\n'
        '- [Shipped model evaluation](model_evaluation_report.md)\n'
        '- [Full production metrics](metrics.json)\n'
        '- [Per-patient test comparison](baseline_tuned_dice.csv)\n'
        '- [Tuned classical baseline](baseline_tuned_report.md)\n'
        '- [Deployment verification](publication_checks/report.md)\n'
        '- [Optimization experiments](experiments/recall_optimization_report.md)\n\n'
        'The shipped U-Net mean test Dice is **0.8139**, measured on 22 previously inspected '
        'test patients. Development-study results concern different patients and are not new '
        'test scores. See the individual reports for tradeoffs and limitations.\n\n'
        '`manifest.json` records original and exported hashes. Historical source fingerprints '
        'describe the corresponding completed runs, not a promise that later source edits '
        'have identical bytes. Refresh with `python scripts/export_reports.py`.\n', encoding='utf-8')
    print(f'Exported {len(manifest)} evidence files, {sum(v["bytes"] for v in manifest):,} bytes.')


if __name__ == '__main__':
    main()
