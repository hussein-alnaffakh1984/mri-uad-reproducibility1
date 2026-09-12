# Environment

Two runtimes are involved, and this file records both rather than presenting one
as if it covered everything.

## 1. The main prediction file

`results/predictions_brain.csv` was produced on 24 July 2026. The versions of the
packages that affect its numbers are recorded inside
`results/predictions_brain_meta.json` and are:

    Python        3.12
    numpy         2.0.2
    scikit-learn  1.6.1
    scipy         1.16
    torch         with the public DINOv2 and torchvision weights

Every AUROC, confidence interval, p-value and figure in Tables 3 to 6 derives
from that file.

## 2. The comparator runs

`predictions_padim.csv`, `predictions_patchcore.csv`, `predictions_dpmm_sens.csv`
and `predictions_anomalydino.csv` were produced later, on the Colab image current
at the time. `requirements.txt` in this folder is a full `pip freeze` of that
image. The packages that affect those numbers are:

    numpy         2.1.3
    scikit-learn  1.6.1
    scipy         1.16.3
    torch         2.11.0+cu128
    torchvision   0.26.0+cu128

The minor-version difference in numpy between the two runtimes does not enter the
comparison. Each comparator is scored against the same stored test set, and every
paired difference reported in Tables 7, 8 and 9 is computed inside a single
analysis pass over both files, so the two runtimes never contribute to opposite
sides of the same subtraction.

## 3. Reproducing

The comparator scripts in `scripts/` read the cached descriptors and write the
prediction files. `scripts/08_seed_averaged_analysis.py` reproduces Tables 3 and 4
from `results/predictions_brain.csv` alone and needs only numpy, pandas and scipy.
