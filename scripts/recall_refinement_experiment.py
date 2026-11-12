"""Validation-only, matched-model study of seeded probability-region growth."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
STUDY = ROOT / 'outputs/experiments/recall_refinement'
LOWS = [0.30, 0.40, 0.50, 0.60, 0.70]


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smoke():
    import numpy as np
    from src.refinement import seeded_region_growth
    from scipy import ndimage as ndi
    rng = np.random.default_rng(42)
    for low in LOWS:
        prob = rng.random((5, 32, 32))
        prob[0] *= 0.69
        result = seeded_region_growth(prob, low)
        strong = prob >= 0.70
        assert (result | strong == result).all()
        assert not (result & (prob < low)).any()
        assert np.array_equal(result.any(axis=(1, 2)), strong.any(axis=(1, 2)))
        oracle = np.stack([ndi.binary_propagation(p >= .7, structure=np.ones((3, 3)), mask=p >= low) for p in prob])
        assert np.array_equal(result, oracle)
        if low == .7:
            assert np.array_equal(result, strong)
    image = np.zeros((8, 8)); image[2, 2] = .8; image[2, 3] = .5; image[6, 6] = .5
    result = seeded_region_growth(image, .4)
    assert result.sum() == 2 and result[2, 3] and not result[6, 6]
    print('SYNTHETIC REFINEMENT SMOKE PASSED', flush=True)


def aggregate(table):
    tp, fp, fn = (int(table[c].sum()) for c in ['true_positive_pixels', 'false_positive_pixels', 'false_negative_pixels'])
    return dict(mean_dice=float(table.dice.mean()), median_dice=float(table.dice.median()),
                pooled_precision=tp/(tp+fp) if tp+fp else 0., pooled_recall=tp/(tp+fn),
                false_alarms=int(table.false_alarms.sum()), tumor_free_slices=int(table.tumor_free_slices.sum()),
                missed_slices=int(table.completely_missed_positive_slices.sum()),
                zero_overlap_slices=int(table.zero_overlap_positive_slices.sum()),
                false_positive_pixels=fp, false_negative_pixels=fn)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    smoke()
    if args.synthetic:
        return
    import numpy as np
    import pandas as pd
    import torch
    from src.config import Config
    from src.baseline_tuned import patient_files
    from src.data import read_mask
    from src.evaluate import predict_probabilities
    from src.model import load_model
    from src.refinement import seeded_region_growth
    from src.utils import seed_everything, device, segmentation_metrics, write_json
    STUDY.mkdir(parents=True, exist_ok=True)
    completed = STUDY / 'completed.json'
    if completed.exists():
        print('Completed validation study already exists; see report.md. No reevaluation.')
        return
    seed_everything(42)
    splits = json.loads((ROOT/'outputs/split.json').read_text())
    val = splits['val']; assert len(val) == 14 and set(val).isdisjoint(splits['test'])
    jobs = json.loads((ROOT/'outputs/experiments/normalization_multiseed/jobs.json').read_text())
    models = {'shipped': ROOT/'checkpoints'}
    models.update({f'seed{seed}': ROOT/jobs[f'whole_volume_seed{seed}']['checkpoint_dir'] for seed in [42, 43, 44]})
    protected = [ROOT/'checkpoints/best.pt', ROOT/'checkpoints/inference.json']
    protected += [p for p in (ROOT/'outputs').iterdir() if p.is_file() and p.suffix in {'.json', '.csv'}]
    protocol_path = STUDY/'protocol.json'
    protocol = dict(low_thresholds=LOWS, high_threshold=.70, validation_patients=val,
                    test_patients_evaluated=0, checkpoint_sha256={name:fingerprint(path/'best.pt') for name,path in models.items()},
                    selection='Choose one shared low threshold using the three corrected-seeding validation models only. Shipped model is a separate descriptive validation check.',
                    gates=dict(min_mean_pooled_recall_gain=.01, min_each_seed_recall_gain=0., max_mean_dice_loss=.002,
                               max_each_seed_dice_loss=.005, max_each_patient_dice_loss=.05,
                               max_mean_precision_loss=.02, max_false_alarm_increase=0, max_missed_slice_increase=0),
                    ranking='Among eligible settings maximize mean Dice, then recall, then larger low threshold. If none qualify retain high-threshold-only inference.',
                    limitations='Same 14 reused validation patients; three training seeds are not independent patient cohorts. No deployment or test evaluation.',
                    protected_sha256={str(p):fingerprint(p) for p in protected})
    if protocol_path.exists():
        frozen = json.loads(protocol_path.read_text())
        assert frozen == protocol, 'Protocol or protected artifacts changed; use a new study directory.'
    else:
        write_json(protocol_path, protocol)
    config = Config()
    frame = patient_files(config, val)  # Never enumerate test patient directories.
    dev = device()
    records = []
    for name, path in models.items():
        csv_path = STUDY/f'{name}_patient_metrics.csv'
        if csv_path.exists():
            records.extend(pd.read_csv(csv_path).to_dict('records')); continue
        run_config = Config(checkpoint_dir=path, output_dir=STUDY/name, prediction_threshold=.70)
        model, checkpoint = load_model(run_config, dev)
        assert checkpoint['splits'] == splits
        assert checkpoint['checkpoint_sha256'] == protocol['checkpoint_sha256'][name]
        cache = STUDY/name/'probabilities'; cache.mkdir(parents=True, exist_ok=True)
        rows = []
        for patient, group in frame.groupby('patient'):
            group = group.sort_values('index')
            signature = [[r.image, Path(r.image).stat().st_size, Path(r.image).stat().st_mtime_ns] for r in group.itertuples()]
            assert signature == checkpoint['normalization'][patient]['signature']
            truth = np.stack([read_mask(r.mask) for r in group.itertuples()]).astype(bool)
            cached = cache/f'{patient}.npz'
            if cached.exists():
                prob = np.load(cached)['probabilities']
            else:
                prob = predict_probabilities(model, group, checkpoint['normalization'], run_config, dev)
                np.savez_compressed(cached, probabilities=prob)
            strong = prob >= .70
            for low in LOWS:
                pred = seeded_region_growth(prob, low)
                assert np.array_equal(pred.any(axis=(1, 2)), strong.any(axis=(1, 2)))
                rows.append(dict(model=name, patient=patient, low=low, **segmentation_metrics(pred, truth)))
            print(f'{name}: measured {patient}', flush=True)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        records.extend(rows)
        del model
        if dev.type == 'cuda': torch.cuda.empty_cache()
    table = pd.DataFrame(records)
    table.to_csv(STUDY/'validation_patient_metrics.csv', index=False)
    summary = pd.DataFrame([dict(model=name, low=float(low), **aggregate(g)) for (name,low),g in table.groupby(['model','low'])])
    summary.to_csv(STUDY/'validation_summary.csv', index=False)
    candidates = []
    for low in LOWS[:-1]:
        gains = []
        worst_patient = 0.
        for name in ['seed42','seed43','seed44']:
            base = summary[(summary.model==name)&(summary.low==.7)].iloc[0]
            cand = summary[(summary.model==name)&(summary.low==low)].iloc[0]
            gains.append({k:float(cand[k]-base[k]) for k in ['mean_dice','pooled_recall','pooled_precision','false_alarms','missed_slices']})
            paired = table[table.model==name].pivot(index='patient',columns='low',values='dice')
            worst_patient = min(worst_patient, float((paired[low]-paired[.7]).min()))
        average = {k:float(np.mean([g[k] for g in gains])) for k in gains[0]}
        eligible = (average['pooled_recall'] >= .01 and min(g['pooled_recall'] for g in gains)>=0
                    and average['mean_dice'] >= -.002 and min(g['mean_dice'] for g in gains)>=-.005
                    and worst_patient >= -.05 and average['pooled_precision'] >= -.02
                    and max(g['false_alarms'] for g in gains)<=0 and max(g['missed_slices'] for g in gains)<=0)
        candidates.append(dict(low=low, eligible=eligible, worst_patient_dice_change=worst_patient,
                               **{f'mean_change_{k}':v for k,v in average.items()}, per_seed=gains))
    eligible = [c for c in candidates if c['eligible']]
    selected = max(eligible, key=lambda c:(c['mean_change_mean_dice'],c['mean_change_pooled_recall'],c['low']))['low'] if eligible else .70
    decision = dict(selected_low=selected, high=.70, passes_prespecified_gates=bool(eligible), candidates=candidates,
                    test_patients_evaluated=0, shipped_model_changed=False,
                    deployment_status='Not deployed; development-only candidate' if eligible else 'No qualifying candidate; retain original inference')
    write_json(STUDY/'selection.json', decision)
    lines = ['# Recall refinement experiment','',
             'The experiment grows 8-connected low-confidence regions only when they contain a probability >=0.70 seed in the same slice. The selected low threshold uses the three completed corrected-seeding whole-volume models on 14 validation patients. The shipped model is reported separately and does not select the setting. The protocol was written before candidate measurements. No test images or masks were accessed.','',
             f'**Selected low threshold: {selected:.2f}; high threshold: 0.70. Gates passed: {bool(eligible)}.** The shipped model and demo remain unchanged.','',
             'The acceptance gates require >=1 percentage-point average pooled recall gain, no recall loss in any seed, <=0.002 average Dice loss, <=0.005 Dice loss in any seed, <=0.05 Dice loss for every patient/seed, <=0.02 average pooled precision loss, and no additional false-alarm or completely missed slices.','',
             '| Model | Low | Mean Dice | Pooled precision | Pooled recall | False alarms | Missed annotated slices |','|---|---:|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.model} | {r.low:.2f} | {r.mean_dice:.4f} | {r.pooled_precision:.2%} | {r.pooled_recall:.2%} | {r.false_alarms}/{r.tumor_free_slices} | {r.missed_slices} |')
    lines += ['', '## What this does and does not resolve','',
              'Slice positivity is exactly preserved by construction. Pixel false positives may increase, which is why precision and Dice are gated. Completely missed slices cannot be rescued by this method. Recovery of those slices requires another experiment, such as a controlled recall-weighted training loss.','',
              'Validation selection is exploratory on repeatedly used development patients. Matched training seeds do not provide an independent cohort. Any candidate needs independent patient validation before claims of generalization. No checkpoint has been substituted and the shipped 0.70 threshold and prior reported numbers are unchanged.','']
    shipped_base = summary[(summary.model=='shipped')&(summary.low==.70)].iloc[0]
    shipped_candidate = summary[(summary.model=='shipped')&(summary.low==selected)].iloc[0]
    changes = {key:float(shipped_candidate[key]-shipped_base[key]) for key in
               ['mean_dice','pooled_precision','pooled_recall','false_alarms','missed_slices']}
    write_json(STUDY/'shipped_candidate_check.json', dict(low=selected,high=.70,
               shipped_validation_changes=changes, passes_same_dice_tolerance=changes['mean_dice']>=-.002,
               recommendation='Use only as a development tradeoff candidate; do not enable if the shipped-model Dice tolerance fails. No retuning using shipped results.'))
    lines += ['## Shipped-model validation check','',
              f"The selected setting changes shipped-model validation Dice by {changes['mean_dice']:+.6f}, pooled recall by {100*changes['pooled_recall']:+.2f} percentage points, and pooled precision by {100*changes['pooled_precision']:+.2f} points. False-alarm and missed-slice counts are unchanged. Passes the same 0.002 Dice-loss tolerance: {changes['mean_dice']>=-.002}. The setting was not retuned using these results.",'']
    for p,h in protocol['protected_sha256'].items(): assert fingerprint(p)==h, p
    (STUDY/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(completed, dict(**decision, protected_artifacts_unchanged=True))
    print(json.dumps(decision,indent=2),flush=True)


if __name__ == '__main__':
    main()
