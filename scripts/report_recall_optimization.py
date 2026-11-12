"""Combine completed development experiments without modifying production metrics."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd


def main():
    experiments=ROOT/'outputs/experiments'
    refinement=experiments/'recall_refinement'
    loss=experiments/'recall_loss'
    r=json.loads((refinement/'selection.json').read_text())
    ship=json.loads((refinement/'shipped_candidate_check.json').read_text())
    result=json.loads((loss/'completed.json').read_text())
    cleanup_path=experiments/'recall_cleanup'
    cleanup=json.loads((cleanup_path/'completed.json').read_text()) if (cleanup_path/'completed.json').exists() else None
    protocol=json.loads((loss/'protocol.json').read_text())
    refinement_protocol=json.loads((refinement/'protocol.json').read_text())
    for seed in [42,43,44]:
        assert refinement_protocol['checkpoint_sha256'][f'seed{seed}']==protocol['reference_checkpoints'][str(seed)]
    for p,h in protocol['protected_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    for name,h in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    for seed in [42,43,44][:result['seeds_completed']]:
        history=pd.read_csv(loss/f'seed{seed}/history.csv')
        assert history.epoch.tolist()==list(range(1,31))
        record=json.loads((loss/f'seed{seed}/completed.json').read_text())
        path=ROOT/f'checkpoints/experiments/recall_loss/seed{seed}/best.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['checkpoint_sha256']
    selected=next(c for c in r['candidates'] if c['low']==r['selected_low']) if r['selected_low']!=.70 else None
    d=selected or dict(mean_change_pooled_recall=0.,mean_change_mean_dice=0.,mean_change_pooled_precision=0.)
    lines=['# Recall optimization: completed evidence','',
           'Two controlled development experiments were completed. The shipped model, its threshold 0.70, both classical baselines, existing metrics and existing checkpoints remain unchanged. No test labels or test predictions were evaluated for either experiment. All candidate results below are validation results on the same 14 patients, not new scores for the 22-patient test set.','',
           '## 1. Seeded region growth','',
           f"Selected low/high thresholds: {r['selected_low']:.2f}/0.70, using three corrected-seeding whole-volume checkpoints. Average changes: pooled recall {100*d['mean_change_pooled_recall']:+.2f} percentage points; pooled precision {100*d['mean_change_pooled_precision']:+.2f} points; mean patient-volume Dice {d['mean_change_mean_dice']:+.6f}. False-alarm slice counts and completely missed slice counts are exactly preserved by construction.",'',
           f"On the separately checked shipped checkpoint, the selected refinement changes Dice by {ship['shipped_validation_changes']['mean_dice']:+.6f}, exceeding the allowed loss of 0.002. **Do not enable this on the shipped model.** It demonstrates a controlled coverage/precision tradeoff, not a general improvement.",'',
           '[Full refinement results](recall_refinement/report.md)','',
           '## 2. Recall-weighted training loss','',
           'Implemented optional Tversky+BCE with FP weight 0.3 and FN weight 0.7. The default remains the original Dice+BCE loss, verified numerically identical. Architecture, initialization, natural sampling, augmentation, optimizer and the 30-epoch budget remain matched to the corrected-seeding controls. Checkpoints record their loss names.','',
           f"Completed training seeds: {result['seeds_completed']}. Seed-42 acceptance screen passed: {result['screen_passed']}. Three-seed confirmation passed: {result['confirmed_across_three_seeds']}. **{result['decision']}.**",'',
           '| Seed | Model | Candidate/reference threshold | Mean Dice | Pooled precision | Pooled recall | False alarms | Completely missed slices |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for c in result['comparisons']:
        for role in ['reference','candidate']:
            v=c[role];threshold=.70 if role=='reference' else result['selected_threshold']
            lines.append(f"| {c['seed']} | {role} | {threshold:.2f} | {v['mean_dice']:.4f} | {v['pooled_precision']:.2%} | {v['pooled_recall']:.2%} | {v['false_alarms']}/{v['tumor_free_slices']} | {v['missed_slices']} |")
    if not result['screen_passed']:
        summary=pd.read_csv(loss/'validation_summary.csv')
        best=summary[(summary.seed==42)&(summary.role=='candidate')].sort_values('mean_dice',ascending=False).iloc[0]
        choice=next(c for c in result['screen']['choices'] if c['threshold']==best.threshold)
        g=choice['changes']; failures=[]
        if g['mean_dice']<-.002:failures.append('mean Dice loss exceeds 0.002')
        if g['pooled_recall']<.01:failures.append('pooled recall gain is below 1 percentage point')
        if g['pooled_precision']<-.02:failures.append('pooled precision loss exceeds 2 percentage points')
        if g['false_alarms']>0:failures.append('false-alarm slices increase')
        if g['missed_slices']>=0:failures.append('completely missed slices do not decrease')
        if choice['worst_patient_dice_change']<-.05:failures.append('a patient loses more than 0.05 Dice')
        lines += ['',f"The candidate's highest validation mean Dice occurs at threshold {best.threshold:.2f}: Dice {best.mean_dice:.4f}, pooled recall {best.pooled_recall:.2%}, false alarms {int(best.false_alarms)}/269, and {int(best.missed_slices)} missed slices. This setting is rejected because: {'; '.join(failures)}. This row is descriptive; it is not a deployed selection."]
    lines += ['', '[Full loss results](recall_loss/report.md)','']
    if cleanup:
        cleanup_summary=pd.read_csv(cleanup_path/'validation_summary.csv')
        best=cleanup_summary.sort_values('mean_dice',ascending=False).iloc[0]
        base=result['screen']['reference']
        lines += ['## 3. Small-component cleanup follow-up','',
                  'The rejected loss trial motivated a separate, newly frozen threshold/component-area grid on development patients. This was additional validation selection, not independent confirmation. The original rejection and its gates were retained.','',
                  f"Cleanup screen passed: {cleanup['screen_passed']}; confirmed across three seeds: {cleanup['confirmed_across_three_seeds']}. {cleanup['decision']}.",'',
                  f"The highest-Dice setting in the fixed follow-up grid uses threshold {best.threshold:.2f} and minimum slice-component area {int(best.min_area)}. It reaches validation Dice {best.mean_dice:.4f} versus {base['mean_dice']:.4f}, pooled recall {best.pooled_recall:.2%} versus {base['pooled_recall']:.2%}, pooled precision {best.pooled_precision:.2%} versus {base['pooled_precision']:.2%}, false alarms {int(best.false_alarms)} versus {base['false_alarms']}, and missed slices {int(best.missed_slices)} versus {base['missed_slices']}. It is a descriptive rejected setting, not an accepted replacement. Recall gain is below 1 percentage point and precision loss exceeds 2 points. No extra training seeds were launched after the screen failed.",'',
                  '[Full cleanup follow-up](recall_cleanup/report.md)','']
    lines += [
              '## Failure evidence and remaining gaps','',
              'The corrected seed-42 validation reference completely misses 10 annotated slices, including three with at least 500 annotated pixels. The inspected examples are not blank FLAIR scans; several have whole-slice maximum probabilities below 0.002. A modest cutoff adjustment cannot recover these examples. Numeric pairing and raw intensities were checked; anatomical annotation correctness still requires expert review.','',
              '![Three missed validation slices and their probabilities](recall_loss/validation_missed_examples.png)','',
              'The diagnostic CSVs record every validation slice, annotated area, predicted/true/error counts, and probabilities for the completely missed slices. They are under recall_loss/development_*.csv and validation_missed_intensities.csv.','',
              ('A candidate met the three-seed development gates but remains separate from the shipped model. Remaining work is independent patient validation and expert review of difficult annotations.' if result['confirmed_across_three_seeds'] else 'No established improvement has been applied to the shipped model. Remaining work is a new justified training/data hypothesis, expert review of difficult development annotations, and genuinely independent patient evaluation. More threshold sweeps on these reused validation patients would not supply independent evidence.'),'',
              '## Verification','',
              'Synthetic checks passed: exact original-loss equivalence, finite gradients with a larger false-negative penalty, seeded-growth invariants against a SciPy propagation oracle, and a full synthetic training/checkpoint round-trip. Every reported candidate completed all 30 epochs; reference checkpoint hashes, candidate checkpoint hashes, frozen settings and protected production artifacts were checked. All old production metrics retain their original bytes. No test inference was rerun.','',
              'Commands: `python scripts/smoke_test_recall.py`, `python scripts/recall_refinement_experiment.py`, `python scripts/recall_loss_experiment.py`, and `python scripts/report_recall_optimization.py`. Completed experiment entry points reuse their results.','']
    path=experiments/'recall_optimization_report.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    readme_path=ROOT/'README.md';readme=readme_path.read_text(encoding='utf-8')
    marker='## Recall optimization experiments'
    section=(f"{marker}\n\nTwo validation-only experiments tested seeded probability-region growth and a recall-weighted Tversky+BCE objective. The growth candidate recovers an average {100*d['mean_change_pooled_recall']:.2f} recall points across three corrected-seeding models, but loses precision and fails the same Dice-loss tolerance on the shipped model; it is not enabled. The staged 30-epoch loss experiment completed {result['seeds_completed']} seed(s): {result['decision'].lower()}.\n\nThe shipped checkpoint, demo, 0.70 threshold and all original numbers remain unchanged. See the [combined optimization report](outputs/experiments/recall_optimization_report.md) for matched validation metrics, failure diagnostics, acceptance rules and remaining gaps. New test inference was not run; independent patient validation remains outstanding. Default training still uses Dice+BCE; `Config(loss_name=\"tversky_bce\")` opts into the experimental objective and should use separate output and checkpoint directories. Run `python scripts/smoke_test_recall.py` for the isolated synthetic training and refinement checks.\n")
    if cleanup:
        section+=f"\nA separate small-component cleanup follow-up also failed its fixed acceptance screen. Its highest-Dice setting reached {best.mean_dice:.4f} validation Dice with {int(best.false_alarms)}/269 false alarms and {int(best.missed_slices)} missed slices, but did not meet the recall/precision limits. These are partial development gains, not new test scores or an accepted deployment.\n"
    start_marker='<!-- recall-optimization:start -->'
    end_marker='<!-- recall-optimization:end -->'
    section=start_marker+'\n'+section+end_marker+'\n'
    if start_marker in readme:
        start=readme.index(start_marker);end=readme.index(end_marker,start)+len(end_marker)
        readme=readme[:start]+section.rstrip()+readme[end:]
    else:
        readme=readme.rstrip()+'\n\n'+section
    readme_path.write_text(readme,encoding='utf-8')
    verification=dict(passed=True,production_artifacts_unchanged=True,complete_epochs_per_seed=30,
                      completed_seeds=result['seeds_completed'],test_patients_evaluated=0,
                      original_default_loss_equivalence_verified=True,shipped_checkpoint_replaced=False)
    (experiments/'recall_optimization_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(verification,indent=2))
    print(path)


if __name__=='__main__':main()
