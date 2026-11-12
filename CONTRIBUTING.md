# Contributing

Use the setup instructions in the README and run commands from the project root.
Keep experiments in separate output and checkpoint directories. Preserve the
fixed patient split, checkpoint identity and inference threshold when comparing
reported results. Choose changes on development patients, not the test set.

Before submitting a change:

- Run the relevant synthetic check: `python scripts/smoke_test.py --synthetic`,
  `python scripts/smoke_test_baseline_tuned.py --synthetic`, or
  `python scripts/smoke_test_recall.py`.
- Describe the change, checks performed and limitations of any accuracy claim.
- Keep data, weights, caches, logs and notebook outputs out of the commit.
- Refresh `reports/` only from completed, verified runs and review its manifest.

Use `python scripts/run_notebooks.py 01_explore` to execute a selected notebook.
Running training/evaluation notebooks can regenerate results; it is not a
read-only report viewer. Saved public evidence is available under `reports/`.
