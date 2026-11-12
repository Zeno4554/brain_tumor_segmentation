# Download the LGG MRI dataset

Create a Kaggle account, accept the dataset terms, install the Kaggle CLI with `pip install kaggle`, and configure your Kaggle API credentials using the Kaggle account settings.

From the repository root run

```bash
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p data --unzip
```

Alternatively, open https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation, choose Download, and extract the archive into `data/`. Ensure the final layout is `data/kaggle_3m/<patient>/<patient>_<n>.tif` with corresponding `<patient>_<n>_mask.tif`. Avoid adding an extra nested `kaggle_3m` directory.

On Kaggle, attach this dataset through Add Input. The code also discovers `/kaggle/input/lgg-mri-segmentation/kaggle_3m` automatically.
