# Recall refinement experiment

The experiment grows 8-connected low-confidence regions only when they contain a probability >=0.70 seed in the same slice. The selected low threshold uses the three completed corrected-seeding whole-volume models on 14 validation patients. The shipped model is reported separately and does not select the setting. The protocol was written before candidate measurements. No test images or masks were accessed.

**Selected low threshold: 0.40; high threshold: 0.70. Gates passed: True.** The shipped model and demo remain unchanged.

The acceptance gates require >=1 percentage-point average pooled recall gain, no recall loss in any seed, <=0.002 average Dice loss, <=0.005 Dice loss in any seed, <=0.05 Dice loss for every patient/seed, <=0.02 average pooled precision loss, and no additional false-alarm or completely missed slices.

| Model | Low | Mean Dice | Pooled precision | Pooled recall | False alarms | Missed annotated slices |
|---|---:|---:|---:|---:|---:|---:|
| seed42 | 0.30 | 0.8704 | 84.05% | 87.82% | 2/269 | 10 |
| seed42 | 0.40 | 0.8709 | 84.51% | 87.41% | 2/269 | 10 |
| seed42 | 0.50 | 0.8712 | 84.90% | 87.04% | 2/269 | 10 |
| seed42 | 0.60 | 0.8716 | 85.32% | 86.65% | 2/269 | 10 |
| seed42 | 0.70 | 0.8717 | 85.75% | 86.20% | 2/269 | 10 |
| seed43 | 0.30 | 0.8580 | 81.39% | 85.63% | 10/269 | 8 |
| seed43 | 0.40 | 0.8582 | 81.83% | 85.19% | 10/269 | 8 |
| seed43 | 0.50 | 0.8586 | 82.26% | 84.78% | 10/269 | 8 |
| seed43 | 0.60 | 0.8588 | 82.70% | 84.34% | 10/269 | 8 |
| seed43 | 0.70 | 0.8588 | 83.15% | 83.86% | 10/269 | 8 |
| seed44 | 0.30 | 0.8614 | 83.57% | 87.36% | 4/269 | 10 |
| seed44 | 0.40 | 0.8619 | 84.14% | 86.83% | 4/269 | 10 |
| seed44 | 0.50 | 0.8622 | 84.67% | 86.32% | 4/269 | 10 |
| seed44 | 0.60 | 0.8622 | 85.19% | 85.79% | 4/269 | 10 |
| seed44 | 0.70 | 0.8619 | 85.73% | 85.21% | 4/269 | 10 |
| shipped | 0.30 | 0.8634 | 81.00% | 87.45% | 5/269 | 9 |
| shipped | 0.40 | 0.8649 | 81.54% | 87.05% | 5/269 | 9 |
| shipped | 0.50 | 0.8660 | 82.03% | 86.67% | 5/269 | 9 |
| shipped | 0.60 | 0.8669 | 82.50% | 86.26% | 5/269 | 9 |
| shipped | 0.70 | 0.8677 | 83.02% | 85.78% | 5/269 | 9 |

## What this does and does not resolve

Slice positivity is exactly preserved by construction. Pixel false positives may increase, which is why precision and Dice are gated. Completely missed slices cannot be rescued by this method. Recovery of those slices requires another experiment, such as a controlled recall-weighted training loss.

Validation selection is exploratory on repeatedly used development patients. Matched training seeds do not provide an independent cohort. Any candidate needs independent patient validation before claims of generalization. No checkpoint has been substituted and the shipped 0.70 threshold and prior reported numbers are unchanged.

## Shipped-model validation check

The selected low threshold was chosen using corrected-seeding models only. On the shipped model, it changes validation mean Dice by -0.002836, pooled recall by +1.27 percentage points, and pooled precision by -1.48 points. False-alarm and missed-slice counts are unchanged. The Dice loss exceeds the same 0.002 tolerance, so this refinement should not be enabled on the shipped checkpoint. The setting was not retuned using these results.
