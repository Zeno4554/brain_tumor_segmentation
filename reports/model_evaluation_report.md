# Evaluation of the shipped U-Net

Assessment: the unchanged U-Net has substantially higher Dice and fewer false alarms than the tuned classical comparator on this split. Its main remaining weakness is missed annotated tissue and uneven performance across patients. These results do not establish performance on independent patients.

Checkpoint: `fda957e3a3cb4b78419099ef9987885648fc35176c59e0dd61bde3873610c247`; epoch 21; threshold **0.70**. This evaluation checks saved measurements; no new inference, training, threshold selection, or baseline evaluation was performed.

| Metric | Shipped U-Net | Tuned FLAIR |
|---|---:|---:|
| Mean patient-volume Dice | 0.8139 | 0.3134 |
| Median patient-volume Dice | 0.8778 | 0.2923 |
| Pooled precision | 0.8446 | 0.2329 |
| Pooled recall | 0.7851 | 0.4646 |
| False alarms / tumor-free slices | 18/483 | 339/483 |
| Zero-prediction annotated slices | 20/295 | 29/295 |
| Zero-overlap patients | 0/22 | 1/22 |

U-Net Dice is higher in 22 patients; tuned FLAIR is higher in 0; ties: 0. This paired comparison is descriptive for this split.

The U-Net misses 21.49% of annotated pixels in aggregate (130,502 pixels). It predicts nothing on 20 annotated slices (6.78%). 2 patients have volume Dice below 0.5.

## Five lowest-scoring patients

| Patient | Dice | Precision | Recall |
|---|---:|---:|---:|
| TCGA_DU_7018_19911220 | 0.4186 | 0.9441 | 0.2689 |
| TCGA_DU_5852_19950709 | 0.4551 | 0.5212 | 0.4039 |
| TCGA_CS_6665_20010817 | 0.5644 | 0.4144 | 0.8848 |
| TCGA_FG_8189_20030516 | 0.6813 | 0.9198 | 0.5410 |
| TCGA_HT_7608_19940304 | 0.7864 | 0.7310 | 0.8509 |

## Institution breakdown

Groups contain only 4-7 patients and are descriptive, not independent site-validation estimates.

| Institution | Patients | Mean Dice |
|---|---:|---:|
| CS | 5 | 0.8125 |
| DU | 6 | 0.7527 |
| FG | 4 | 0.8352 |
| HT | 7 | 0.8552 |

## Interpretation and next validation

Keep this checkpoint as the current project reference. The tuned baseline strengthens the comparison; it did not modify or weaken the U-Net. Improvement work should prioritize recovery of missed annotated tissue while monitoring the existing false-alarm definition on development patients. Freeze any future candidate using development evidence, then use genuinely new patients or a prespecified independent evaluation to assess generalization. Repeatedly scoring these same 22 patients does not create new evidence.

The checkpoint predates the worker-seeding fix. Corrected-seeding runs have validation results, but they are separate candidates and do not replace these shipped-checkpoint numbers.

Checks passed: checkpoint identity, threshold identity, 22 patient records, 778 slice records, patient-volume mean/median, and agreement of pixel/slice totals between saved slice metrics, patient metrics, and metrics.json. Original artifacts and all reported numbers remain unchanged.
