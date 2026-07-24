# Locally Concentrated Patch Aggregation in Training-Free Brain MRI Anomaly Detection

Reproducibility package for the manuscript *An Empirical Analysis of Locally Concentrated
Patch Aggregation in Training-Free Brain MRI Anomaly Detection*, International Journal of
Intelligent Engineering and Systems.

The study holds a frozen backbone, reference set, and scoring rule fixed across three brain
sequences of one fastMRI cohort, and varies only the operator that aggregates per-patch
deviations into a slice score.

## What this package guarantees

Every estimate, confidence interval, and p-value in the manuscript is regenerated from one
stored prediction file by one analysis script, under one patient-level cluster bootstrap and
two pre-declared multiplicity families. No feature extraction is required to reproduce any
table.

The lesion-area statistics of Section 4.2.1 are the sole exception: they derive from the
fastMRI+ annotation boxes, and the per-slice table used to compute them is included as
`data/lesion_area_per_slice.csv`.

## Reproduce the tables

    pip install -r requirements.txt
    python scripts/02_analysis.py --results results/ --out results/regenerated/
    python scripts/03_make_figures.py --results results/ --out figures/regenerated/

Stage 01 is needed only to rebuild the prediction file from scratch and requires the
frozen feature caches; the shipped prediction file makes it unnecessary for reproduction.

Compare `results/regenerated/` against the shipped `results/*.csv`; they are identical under
the fixed seed.

## Layout

    data/       brain_split.csv             per-volume cohort definition
                lesion_area_per_slice.csv   fastMRI+ box areas, 320 x 320 native frame
    results/    predictions_brain.csv       score of every test slice, per configuration and seed
                results_*.csv               effects, interaction, volume level, operating points
                anomalydino_*, dpmm_*       same-split comparator evaluations
                focality_*                  lesion-area stratification (Section 4.2.1)
    scripts/    01_build_predictions.py     builds the prediction file from the frozen feature caches
                02_analysis.py              regenerates every table from the prediction file
                03_make_figures.py          regenerates Figures 2-5 from the result tables
    figures/    figures as published (PNG and vector PDF)
    docs/       DATA.md                     how to obtain fastMRI and fastMRI+

## Prediction file schema

| column | meaning |
|---|---|
| sequence | AXT1POST, AXFLAIR, AXT1 |
| level | image or patch |
| backbone | dino, wrn, ensemble |
| rule | knn, lof, knn+lof |
| operator | cls, max, top1, top5, mean |
| seed | -1 for deterministic image level, 0-4 for patch memory-bank draws |
| volume, slice | patient-level identifier and slice index |
| y_true | 0 normal, 1 abnormal |
| score | anomaly score |

## Reproducibility notes

- All uncertainty is quantified by a patient-level cluster bootstrap (B = 2000) that
  resamples whole volumes; slices are never resampled independently.
- Splits are constructed at the volume level: no acquisition contributes to both the
  reference and the test set.
- Every stochastic component is repeated across five seeds; patch results are reported as
  the mean with its standard deviation.
- Performance columns report the five-seed mean, while paired differences and their
  intervals are computed from the seed-0 prediction set, since a paired bootstrap requires
  matched per-slice predictions. The two differ by at most 0.007 AUROC.

## Data

Neither imaging data nor model weights are redistributed. See `docs/DATA.md`. Encoder
weights are downloaded from their public hubs on first run.

## Scope

This package covers the brain cohort only. An earlier version of the study included a knee
cohort; it was withdrawn during revision and is not part of the published analysis.

## License

MIT.
