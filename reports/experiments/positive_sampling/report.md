# Positive-sampling experiment

This experiment changes training slice sampling to an expected 50% tumor-containing slices and 50% tumor-free slices. Weighted sampling uses replacement and retains 2,710 training draws per epoch. The target fraction is an expectation, so actual epoch proportions vary. Original sampling used each training slice once per shuffled epoch, with 33.43% positive slices.

Both runs use seed 42, ImageNet-initialized ResNet34 U-Net, 30 epochs, batch size 8, the original loss, augmentations, learning rates, scheduler, and the fixed 74/14/22 patient split. Each checkpoint is selected by its own best mean validation-volume Dice. The sampling choice also changes which slices recur and their augmentation exposure.

## Validation results

| Model | Mean volume Dice | Median volume Dice | Mean patient recall | Pooled recall | False alarms | False-alarm rate |
|---|---:|---:|---:|---:|---:|---:|
| candidate | 0.8624 | 0.8931 | 84.63% | 81.65% | 6/269 | 2.23% |
| reference | 0.8660 | 0.8773 | 90.00% | 86.67% | 5/269 | 1.86% |

Mean paired Dice change is -0.0035, with a patient-bootstrap 95% interval from -0.0266 to +0.0148. Dice improved for 10 patients and worsened for 4. The bootstrap uses 10,000 resamples with seed 42 and describes variability among these 14 validation patients. Both runs selected checkpoints on this validation set, and this single-seed experiment does not measure training-seed variability.

## Patient comparison

| Patient | Reference Dice | Candidate Dice | Change |
|---|---:|---:|---:|
| TCGA_CS_5396_20010302 | 0.8266 | 0.7226 | -0.1040 |
| TCGA_HT_A5RC_19990831 | 0.7849 | 0.7082 | -0.0767 |
| TCGA_FG_6689_20020326 | 0.8130 | 0.7878 | -0.0252 |
| TCGA_HT_7694_19950404 | 0.9115 | 0.8930 | -0.0185 |
| TCGA_HT_8107_19980708 | 0.8842 | 0.8924 | +0.0082 |
| TCGA_DU_6404_19850629 | 0.9392 | 0.9478 | +0.0085 |
| TCGA_CS_4942_19970222 | 0.9253 | 0.9358 | +0.0105 |
| TCGA_DU_5851_19950428 | 0.8984 | 0.9103 | +0.0119 |
| TCGA_DU_5874_19950510 | 0.8570 | 0.8719 | +0.0149 |
| TCGA_HT_7690_19960312 | 0.9422 | 0.9578 | +0.0155 |
| TCGA_HT_7693_19950520 | 0.8823 | 0.8996 | +0.0173 |
| TCGA_HT_7884_19980913 | 0.8723 | 0.8931 | +0.0208 |
| TCGA_CS_6669_20020102 | 0.8688 | 0.8937 | +0.0249 |
| TCGA_DU_7010_19860307 | 0.7179 | 0.7599 | +0.0421 |

## Decision from this experiment

Use the reference checkpoint for the demo and reported model results. In this run, the sampling candidate has lower mean validation Dice.

The two largest Dice regressions are TCGA_CS_5396_20010302 and TCGA_HT_A5RC_19990831. Patient gains and losses have different magnitudes, so the number of improved patients alone does not determine the mean. The uncertainty interval reflects resampling of these validation patients. Training-seed variability and checkpoint-selection uncertainty require additional experiments.

### TCGA_CS_5396_20010302

*MRI illustration retained in the local generated report.*

*MRI illustration retained in the local generated report.*

### TCGA_HT_A5RC_19990831

*MRI illustration retained in the local generated report.*

*MRI illustration retained in the local generated report.*

## Evidence and interpretation

The accompanying reference and candidate PNGs show each model's largest-error slice for every validation patient. Their selected slice can differ between models. Green is correct overlap, red is excess segmentation, and blue is missed annotated tumor. `validation_patient_metrics.csv` contains precision, recall, and full-volume pixel counts. `history.csv` records all epochs, and `sampling.json` records the sampling distribution.

The experiment assesses the proposed sampling strategy using validation patients. A higher validation Dice needs to be considered alongside recall and false alarms. Any selected change still needs confirmation across additional seeds and independent patients, since the original test cases have already informed the diagnosis.

Reference checkpoint SHA-256 `fda957e3a3cb4b78419099ef9987885648fc35176c59e0dd61bde3873610c247`.

Candidate checkpoint SHA-256 `56ee017dc235ed7bacf0eeb7b888c7b457e541b8fa209fee1b8896b2cdf61bc9`.
