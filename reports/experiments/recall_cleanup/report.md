# Small-component follow-up

Seed-42 loss trial improved recall and missed slices but increased false positives. This separate development follow-up tests small-component cleanup; it does not change the original loss-study rejection.

A fresh protocol freezes threshold/area grids and acceptance gates before cleanup measurements. Seed 42 screens settings on the reused 14 validation patients; any passing setting is frozen before training matched seeds 43/44. This is additional development selection on a small reused cohort, not independent validation. No test images or masks were used.

**No qualifying follow-up; keep shipped model.** Seed-42 screen passed: False; evaluated seeds: 1.

| Seed | Model | Threshold | Min area | Dice | Precision | Recall | False alarms | Missed slices |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | reference | 0.70 | 0 | 0.8717 | 85.75% | 86.20% | 2/269 | 10 |
| 42 | highest Dice; rejected | 0.80 | 200 | 0.8775 | 83.57% | 86.81% | 2/269 | 9 |

Highest seed-42 Dice among the fixed cleanup grid: 0.8775, threshold 0.80, area 200, precision 83.57%, recall 86.81%, false alarms 2/269, missed slices 9. No setting met every gate; this descriptive row is not an accepted selection.

No deployment was changed. Removing components can erase real small tumors; per-patient Dice, recall and missed-slice gates explicitly check that tradeoff. The loss-only trial remains recorded as a rejected primary screen. New patients and expert review remain required.
