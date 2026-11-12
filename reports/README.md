# Verified results snapshot

These files summarize completed local runs; exporting them does not rerun the test set. Original measurements remain in ignored `outputs/`. Machine-specific paths in copied JSON are made relative. MRI illustrations, raw data, probability arrays and weights stay local.

- [Shipped model evaluation](model_evaluation_report.md)
- [Full production metrics](metrics.json)
- [Per-patient test comparison](baseline_tuned_dice.csv)
- [Tuned classical baseline](baseline_tuned_report.md)
- [Deployment verification](publication_checks/report.md)
- [Optimization experiments](experiments/recall_optimization_report.md)

The shipped U-Net mean test Dice is **0.8139**, measured on 22 previously inspected test patients. Development-study results concern different patients and are not new test scores. See the individual reports for tradeoffs and limitations.

`manifest.json` records original and exported hashes. Historical source fingerprints describe the corresponding completed runs, not a promise that later source edits have identical bytes. Refresh with `python scripts/export_reports.py`.
