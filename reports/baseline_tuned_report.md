# Tuned classical FLAIR baseline

Selected tuned-baseline setting: `k=1.0`, `E=9`, `min_area=200`, `largest_3d=False`. Development mean/median Dice: 0.3282/0.3317; test false alarms: 339/483, pooled precision/recall: 23.29%/46.46%, zero-prediction annotated slices: 29/295, zero-overlap patients: 1/22. Development and test errors are tabulated below.

The 440-setting grid was selected only on 74 train + 14 validation patients. Configuration frozen at 2026-11-12T08:43:38.044527+00:00, before accessing test directories. The fixed comparator was then evaluated once on the 22 test patients; completed patient predictions and measurements are cached.

raw FLAIR index 1; filled >10 mask; largest 8-connected slice component; Euclidean disk erosion; patient-pooled float64 population mean/std; slice 8-connectivity; volume 26-connectivity.

exact mean Dice ties: smaller E, smaller min_area, no 3D filter, then smaller k.

The selected erosion and minimum area are at the upper bounds of the prescribed grid. This identifies the best tested setting, not a global optimum; the grid was not expanded after seeing test results.

This is a descriptive comparison on one split of 22 previously inspected patients. The U-Net margin does not establish a general result. The shipped checkpoint, threshold 0.70, spec baseline and all previous metrics are unchanged.

## Development versus test

| Metric | Development (88) | Test (22) |
|---|---:|---:|
| mean_dice | 0.328248 | 0.313424 |
| median_dice | 0.331669 | 0.292330 |
| false_alarms | 1440 | 339 |
| tumor_free_slices | 2073 | 483 |
| false_alarm_rate | 0.694645 | 0.701863 |
| pooled_precision | 0.255905 | 0.232920 |
| pooled_recall | 0.608101 | 0.464606 |
| true_positive_pixels | 1243337 | 282123 |
| false_positive_pixels | 3615260 | 929119 |
| false_negative_pixels | 801286 | 325108 |
| tumor_containing_slices | 1078 | 295 |
| completely_missed_positive_slices | 60 | 29 |
| zero_overlap_positive_slices | 188 | 76 |
| patients_with_zero_overlap | 2 | 1 |

False alarms mean any predicted pixel on a tumor-free slice. A completely missed positive slice has no predicted pixels anywhere; zero-overlap slices may contain off-target predictions. Dice uses the same function as the U-Net: exact zero for nonempty zero-overlap volumes and one for two empty volumes.

## Patient-volume comparison

| Patient | Tuned FLAIR Dice | Shipped U-Net Dice (0.70) |
|---|---:|---:|
| TCGA_CS_4941_19960909 | 0.271096 | 0.867893 |
| TCGA_CS_5393_19990606 | 0.695828 | 0.818769 |
| TCGA_CS_6290_20000917 | 0.302518 | 0.922213 |
| TCGA_CS_6665_20010817 | 0.392570 | 0.564402 |
| TCGA_CS_6666_20011109 | 0.279189 | 0.889145 |
| TCGA_DU_5852_19950709 | 0.141966 | 0.455110 |
| TCGA_DU_6407_19860514 | 0.355814 | 0.930339 |
| TCGA_DU_7018_19911220 | 0.405928 | 0.418557 |
| TCGA_DU_7301_19911112 | 0.140923 | 0.920593 |
| TCGA_DU_7306_19930512 | 0.051017 | 0.895492 |
| TCGA_DU_A5TS_19970726 | 0.099339 | 0.895896 |
| TCGA_FG_6688_20020215 | 0.456611 | 0.873887 |
| TCGA_FG_6692_20020606 | 0.128328 | 0.877237 |
| TCGA_FG_7637_20000922 | 0.275181 | 0.908219 |
| TCGA_FG_8189_20030516 | 0.000000 | 0.681265 |
| TCGA_HT_7602_19951103 | 0.441613 | 0.802371 |
| TCGA_HT_7605_19950916 | 0.287491 | 0.878362 |
| TCGA_HT_7608_19940304 | 0.190656 | 0.786374 |
| TCGA_HT_7616_19940813 | 0.297169 | 0.789873 |
| TCGA_HT_7686_19950629 | 0.607910 | 0.915235 |
| TCGA_HT_7882_19970125 | 0.656683 | 0.885727 |
| TCGA_HT_A61A_20000127 | 0.417507 | 0.928452 |

## Figures

Each figure uses the largest annotated slice in numeric order. Green is truth, blue is tuned FLAIR, and red is the unchanged shipped U-Net at 0.70.

*MRI illustration retained in the local generated report.*

*MRI illustration retained in the local generated report.*

## Verification and run history

The initial serial attempt was stopped during development-only searching to enable three CPU threads. The threaded search restarted from the beginning and completed every patient and every grid setting. No test evaluation occurred during the interrupted attempt.

The frozen configuration and grid hashes, development objective, 22 cached test receipts, and exact U-Net file-list correspondence were checked. SHA-256 checks confirm the protected artifacts and source files are unchanged; all pre-existing metrics keys retain their exact values. The isolated synthetic smoke test checks the full grid and independent SciPy morphology/component oracles.

Artifacts: `baseline_tuned_search.csv`, `baseline_tuned_config.json`, `baseline_tuned_development_dice.csv`, `baseline_tuned_dice.csv`, `baseline_tuned_test_files.json`, `baseline_tuned_preservation.json`, and the `baseline_tuned` section of `metrics.json`.
