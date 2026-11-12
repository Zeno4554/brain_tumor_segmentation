# Local model checkpoints

Model weights and inference metadata remain local and are excluded from Git.
The documented shipped model expects `best.pt` and its matching `inference.json`
here. Experiments use separate subdirectories.

The original checkpoint SHA-256 and threshold are recorded in
[`reports/deployment_manifest.json`](../reports/deployment_manifest.json).
Cloning this repository does not download weights. Train a model before starting
the demo, or restore the original verified weights and matching metadata from
your own artifact storage. A fresh training run need not reproduce the historical
checkpoint or its reported numbers.
