# Project structure

```text
brain-tumor-segmentation/
├── .vscode/extensions.json      # Optional editor recommendations
├── checkpoints/README.md       # Local weights instructions; binaries ignored
├── data/README.md              # Local dataset instructions; scans ignored
├── docs/                       # Repository and contribution guidance
├── notebooks/                  # Five ordered notebooks, without saved outputs
├── reports/                    # Curated, portable snapshots of verified evidence
│   ├── experiments/            # Completed development studies
│   ├── publication_checks/     # Verification evidence
│   └── README.md               # Results index and snapshot policy
├── scripts/                    # Experiments, checks and maintenance entry points
├── src/                        # Data, model, training, inference and analysis
├── outputs/                    # Local generated runs, caches and executed notebooks
├── .editorconfig
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

`src/` is the existing importable package (`python -m src.train`, for example).
Its name and the existing data/checkpoint/output locations are retained so
notebooks and command-line tools continue to work.

`outputs/` is ignored. `reports/` contains small result tables, portable JSON
metadata, Markdown reports and aggregate plots exported from completed runs.
Raw MRI images, probability arrays, model weights, logs, caches and embedded
notebook outputs are not included in that snapshot. MRI illustrations remain
available locally in the original reports and executed notebook archive.

Refresh the public snapshot with `python scripts/export_reports.py`. This does
not run inference, change measurements or overwrite local experimental results.
`reports/manifest.json` records the source and destination hashes. Machine paths
in copied JSON are converted to project-relative paths; original files stay intact.

## GitHub preparation

The project was found under an ancestor Git repository rooted at `C:/`.
Run Git commands from this project directory and create its own repository
before staging anything:

```powershell
Set-Location C:\projects\ML\brain-tumor-segmentation
git init -b main
git rev-parse --show-toplevel
git status --short
```

The reported top-level directory must be this project, not `C:/`.
Review the files before committing. No remote, commit, push or source-code license
has been selected by the cleanup. Dataset terms remain those of its source.

Executed notebooks are backed up under ignored `outputs/notebook_archive/`
before their outputs are cleared. Keep any checkpoints you need in separate
artifact storage; Git is not the backup for ignored local files.
