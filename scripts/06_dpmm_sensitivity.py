"""Dirichlet-process mixture comparator and its sensitivity grid, Table 8.

Answers Reviewer 2 comment 3. Nine settings: three reduced dimensionalities
(64, 128, 256) against three truncation levels (20, 50, 100). The setting used
in the manuscript, PCA 128 with truncation 50, is run over three seeds; the
remaining settings over one, since the grid is a sensitivity check and not an
estimate.

A component is counted as retained when its mixture weight exceeds 0.01; the
count depends on that threshold and the threshold is therefore stated.
"""
import os, time, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import roc_auc_score

CACHE = "/content/drive/MyDrive"   # adjust if the caches live elsewhere
OUT   = "/content/drive/MyDrive/uad_results/predictions_dpmm_sens.csv"
SEQS  = {"AXT1POST": "brain_cache_axt1post.npz",
         "AXFLAIR":  "brain_cache_axflair.npz",
         "AXT1":     "brain_cache_axt1.npz"}
FIT_N, ANCHOR, W_KEEP = 20000, (128, 50), 0.01
GRID = [(p, t) for p in (64, 128, 256) for t in (20, 50, 100)]
OURS = {"AXT1POST": 0.8732, "AXFLAIR": 0.8701, "AXT1": 0.8228}


def l2(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


def top5(pm):
    thr = np.quantile(pm, 0.95, axis=1, keepdims=True)
    m = pm >= thr
    return (pm * m).sum(1) / np.maximum(m.sum(1), 1)


rows, summary = [], []
for seq, fn in SEQS.items():
    z   = np.load(os.path.join(CACHE, fn))
    ref = l2(z["ref_dpatch"].astype(np.float32))
    qa  = l2(np.concatenate([z["tn_dpatch"], z["ta_dpatch"]], 0).astype(np.float32))
    y   = np.r_[np.zeros(len(z["tn_dpatch"]), int), np.ones(len(z["ta_dpatch"]), int)]
    vol = np.r_[z["tn_vol"],   z["ta_vol"]]
    sl  = np.r_[z["tn_slice"], z["ta_slice"]]
    N, P, _ = qa.shape
    pool = ref.reshape(-1, ref.shape[2])
    print("==", seq, "pool", pool.shape, flush=True)

    for pca_d, trunc in GRID:
        for seed in ([0, 1, 2] if (pca_d, trunc) == ANCHOR else [0]):
            t0 = time.time()
            rng = np.random.default_rng(seed)
            sub = pool[np.sort(rng.choice(len(pool), min(FIT_N, len(pool)), replace=False))]
            pca = PCA(n_components=pca_d, random_state=seed).fit(sub)
            g = BayesianGaussianMixture(
                    n_components=trunc, covariance_type="diag",
                    weight_concentration_prior_type="dirichlet_process",
                    max_iter=200, random_state=seed).fit(pca.transform(sub))
            keep = np.where(g.weights_ > W_KEEP)[0]
            mu, va = g.means_[keep], np.maximum(g.covariances_[keep], 1e-6)
            Qp = pca.transform(qa.reshape(-1, qa.shape[2])).astype(np.float32)
            d = np.empty(len(Qp), np.float32)
            for i in range(0, len(Qp), 20000):
                c = Qp[i:i + 20000][:, None, :]
                d[i:i + 20000] = np.sqrt((((c - mu[None]) ** 2) / va[None]).sum(2)).min(1)
            s = top5(d.reshape(N, P))
            a = roc_auc_score(y, s)
            summary.append(dict(sequence=seq, pca=pca_d, trunc=trunc, seed=seed,
                                kept=len(keep), auroc=a, gap=OURS[seq] - a))
            rows += [dict(sequence=seq, level="patch",
                          backbone="dpmm_p%d_t%d" % (pca_d, trunc), rule="dpmm",
                          operator="top5", seed=seed, volume=str(v), slice=int(x),
                          y_true=int(t), score=float(c))
                     for v, x, t, c in zip(vol, sl, y, s)]
            print("   pca=%-3d trunc=%-3d seed=%d kept=%-3d AUROC=%.4f gap=%+.4f (%.0fs)"
                  % (pca_d, trunc, seed, len(keep), a, OURS[seq] - a, time.time() - t0),
                  flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
S = pd.DataFrame(summary)
S.to_csv(OUT.replace(".csv", "_summary.csv"), index=False)
print(S.pivot_table(index=["sequence", "pca"], columns="trunc",
                    values="gap", aggfunc="mean").round(4))
