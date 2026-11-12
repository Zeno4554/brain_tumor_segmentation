# Recall optimization: completed evidence

Two controlled development experiments were completed. The shipped model, its threshold 0.70, both classical baselines, existing metrics and existing checkpoints remain unchanged. No test labels or test predictions were evaluated for either experiment. All candidate results below are validation results on the same 14 patients, not new scores for the 22-patient test set.

## 1. Seeded region growth

Selected low/high thresholds: 0.40/0.70, using three corrected-seeding whole-volume checkpoints. Average changes: pooled recall +1.39 percentage points; pooled precision -1.39 points; mean patient-volume Dice -0.000467. False-alarm slice counts and completely missed slice counts are exactly preserved by construction.

On the separately checked shipped checkpoint, the selected refinement changes Dice by -0.002836, exceeding the allowed loss of 0.002. **Do not enable this on the shipped model.** It demonstrates a controlled coverage/precision tradeoff, not a general improvement.

[Full refinement results](recall_refinement/report.md)

## 2. Recall-weighted training loss

Implemented optional Tversky+BCE with FP weight 0.3 and FN weight 0.7. The default remains the original Dice+BCE loss, verified numerically identical. Architecture, initialization, natural sampling, augmentation, optimizer and the 30-epoch budget remain matched to the corrected-seeding controls. Checkpoints record their loss names.

Completed training seeds: 1. Seed-42 acceptance screen passed: False. Three-seed confirmation passed: False. **No candidate met the staged confirmation requirements; retain shipped model.**

| Seed | Model | Candidate/reference threshold | Mean Dice | Pooled precision | Pooled recall | False alarms | Completely missed slices |
|---|---|---:|---:|---:|---:|---:|---:|
| 42 | reference | 0.70 | 0.8717 | 85.75% | 86.20% | 2/269 | 10 |
| 42 | candidate | 0.70 | 0.8757 | 82.28% | 88.02% | 6/269 | 6 |

The candidate's highest validation mean Dice occurs at threshold 0.80: Dice 0.8765, pooled recall 87.36%, false alarms 6/269, and 6 missed slices. This setting is rejected because: pooled precision loss exceeds 2 percentage points; false-alarm slices increase. This row is descriptive; it is not a deployed selection.

[Full loss results](recall_loss/report.md)

## 3. Small-component cleanup follow-up

The rejected loss trial motivated a separate, newly frozen threshold/component-area grid on development patients. This was additional validation selection, not independent confirmation. The original rejection and its gates were retained.

Cleanup screen passed: False; confirmed across three seeds: False. No qualifying follow-up; keep shipped model.

The highest-Dice setting in the fixed follow-up grid uses threshold 0.80 and minimum slice-component area 200. It reaches validation Dice 0.8775 versus 0.8717, pooled recall 86.81% versus 86.20%, pooled precision 83.57% versus 85.75%, false alarms 2 versus 2, and missed slices 9 versus 10. It is a descriptive rejected setting, not an accepted replacement. Recall gain is below 1 percentage point and precision loss exceeds 2 points. No extra training seeds were launched after the screen failed.

[Full cleanup follow-up](recall_cleanup/report.md)

## Failure evidence and remaining gaps

The corrected seed-42 validation reference completely misses 10 annotated slices, including three with at least 500 annotated pixels. The inspected examples are not blank FLAIR scans; several have whole-slice maximum probabilities below 0.002. A modest cutoff adjustment cannot recover these examples. Numeric pairing and raw intensities were checked; anatomical annotation correctness still requires expert review.

*MRI illustration retained in the local generated report.*

The diagnostic CSVs record every validation slice, annotated area, predicted/true/error counts, and probabilities for the completely missed slices. They are under recall_loss/development_*.csv and validation_missed_intensities.csv.

No established improvement has been applied to the shipped model. Remaining work is a new justified training/data hypothesis, expert review of difficult development annotations, and genuinely independent patient evaluation. More threshold sweeps on these reused validation patients would not supply independent evidence.

## Verification

Synthetic checks passed: exact original-loss equivalence, finite gradients with a larger false-negative penalty, seeded-growth invariants against a SciPy propagation oracle, and a full synthetic training/checkpoint round-trip. Every reported candidate completed all 30 epochs; reference checkpoint hashes, candidate checkpoint hashes, frozen settings and protected production artifacts were checked. All old production metrics retain their original bytes. No test inference was rerun.

Commands: `python scripts/smoke_test_recall.py`, `python scripts/recall_refinement_experiment.py`, `python scripts/recall_loss_experiment.py`, and `python scripts/report_recall_optimization.py`. Completed experiment entry points reuse their results.
