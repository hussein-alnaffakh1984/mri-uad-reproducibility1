# Files added for the second revision

## Provided here

`comparator_auroc_by_seed.csv` gives every seed-level AUROC of the four
re-implemented comparators, and `predictions_dpmm_sens_summary.csv` gives the
full 3x3 sensitivity grid with the number of retained mixture components. Between
them they reproduce every AUROC reported in Tables 7, 8 and 9 and in the
sensitivity grid of the Table 8 note.

## Still to be copied from the run output

The two files above are summaries. Re-running the paired patient-level bootstrap
needs the per-slice scores, which are in these four files:

    predictions_padim.csv         PaDiM, 5 seeds x 4 operators
    predictions_patchcore.csv     PatchCore, 2 coreset sizes x 5 seeds x 4 operators
    predictions_dpmm_sens.csv     DPMM, 9 settings, top-5% operator
    predictions_anomalydino.csv   AnomalyDINO, 5 seeds, top-1% operator

Each carries the same columns as `predictions_brain.csv`, so any of them can be
concatenated with it and passed to the same analysis code.
