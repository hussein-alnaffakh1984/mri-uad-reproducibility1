"""PaDiM baseline, Table 9.

Training-free. Fits a Gaussian per patch position over a random subset of 100 of
the 768 DINOv2 channels, with a shrinkage term of 1e-3, and scores a patch by its
Mahalanobis distance to that position's mean. Consumes the same cached patch
descriptors, the same reference volumes and the same test split as the main
configuration, and is scored under the same four aggregation operators and the
same five seeds.

Output columns match results/predictions_brain.csv exactly, so the file can be
concatenated with it and passed to the same analysis code.
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

CACHE = "/content/drive/MyDrive"   # adjust if the caches live elsewhere
OUT   = "/content/drive/MyDrive/uad_results/predictions_padim.csv"
SEQS  = {"AXT1POST": "brain_cache_axt1post.npz",
         "AXFLAIR":  "brain_cache_axflair.npz",
         "AXT1":     "brain_cache_axt1.npz"}
D_SEL, EPS, SEEDS = 100, 1e-3, [0, 1, 2, 3, 4]


def agg(pm, op):
    if op == "max":  return pm.max(1)
    if op == "mean": return pm.mean(1)
    q = 0.99 if op == "top1" else 0.95
    thr = np.quantile(pm, q, axis=1, keepdims=True)
    m = pm >= thr
    return (pm * m).sum(1) / np.maximum(m.sum(1), 1)


rows = []
for seq, fn in SEQS.items():
    z   = np.load(os.path.join(CACHE, fn))
    ref = z["ref_dpatch"]
    q   = np.concatenate([z["tn_dpatch"], z["ta_dpatch"]], 0)
    y   = np.r_[np.zeros(len(z["tn_dpatch"]), int), np.ones(len(z["ta_dpatch"]), int)]
    vol = np.r_[z["tn_vol"],   z["ta_vol"]]
    sl  = np.r_[z["tn_slice"], z["ta_slice"]]
    P, Dm = ref.shape[1], ref.shape[2]
    print(seq, "ref", ref.shape, "test", q.shape, flush=True)

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(Dm, D_SEL, replace=False))
        R = ref[:, :, idx].astype(np.float64)
        Q = q[:, :, idx].astype(np.float64)
        pm = np.empty((len(Q), P), np.float32)
        for p in range(P):
            X  = R[:, p, :]
            mu = X.mean(0)
            Ci = np.linalg.inv(np.cov(X, rowvar=False) + EPS * np.eye(D_SEL))
            d  = Q[:, p, :] - mu
            pm[:, p] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, Ci, d), 0))
        for op in ["max", "mean", "top1", "top5"]:
            s = agg(pm, op)
            rows += [dict(sequence=seq, level="patch", backbone="padim",
                          rule="mahalanobis", operator=op, seed=seed,
                          volume=str(v), slice=int(x), y_true=int(t), score=float(c))
                     for v, x, t, c in zip(vol, sl, y, s)]
        print("   seed %d  max %.4f  top5 %.4f" %
              (seed, roc_auc_score(y, agg(pm, "max")), roc_auc_score(y, agg(pm, "top5"))),
              flush=True)
        del R, Q, pm

os.makedirs(os.path.dirname(OUT), exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print("wrote", OUT, len(rows), "rows")
