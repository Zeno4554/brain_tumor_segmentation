# Recall-weighted loss experiment

Candidate training changes only Dice+BCE to Tversky+BCE (FP weight 0.3, FN weight 0.7). Architecture, initialization, optimizer, augmentations, natural slice sampling, whole-volume normalization, checkpoint selection at 0.5, and 30-epoch budget match the completed corrected-seeding controls. Neither inference refinement nor postprocessing is used in this loss comparison.

Seed 42 is screened first. The validation threshold grid and acceptance gates were frozen before training. A candidate must gain >=1 percentage point pooled recall, reduce completely missed positive slices, lose <=0.002 mean Dice and <=0.02 pooled precision, add no false-alarm slices, and lose <=0.05 Dice for any patient. A passing seed-42 threshold is frozen for seeds 43 and 44; otherwise the experiment stops rather than spending two additional training runs on a rejected candidate.

**No candidate met the staged confirmation requirements; retain shipped model.** Screen passed: False; completed seeds: 1; selected candidate threshold: 0.70.

| Seed | Model | Threshold | Mean Dice | Pooled precision | Pooled recall | False alarms | Missed slices |
|---|---|---:|---:|---:|---:|---:|---:|
| 42 | reference | 0.70 | 0.8717 | 85.75% | 86.20% | 2/269 | 10 |
| 42 | candidate | 0.70 | 0.8757 | 82.28% | 88.02% | 6/269 | 6 |

## Interpretation

This is development evidence on the same 14 reused validation patients. A negative screen does not prove the loss can never work; it means this prespecified candidate did not meet the project tradeoff requirements. A positive result would still need independent patient confirmation. The original 22 test patients contribute no labels or predictions to this experiment.

The shipped checkpoint, demo, 0.70 deployment threshold, classical baselines and all existing production numbers remain unchanged. Independent validation cannot be supplied by rerunning the existing test set.

Full threshold results are in validation_summary.csv; patient metrics, frozen protocol, seed histories, checkpoint metadata, and decisions are retained alongside this report.
