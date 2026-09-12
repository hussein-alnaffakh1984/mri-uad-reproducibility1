"""PatchCore baseline, Table 9.

Training-free. Replaces the uniform subsample of the memory bank with the greedy
coreset of the original method, at two bank sizes: 1 and 10 per cent of the
available patch descriptors. Two sizes are reported so that the comparison is
not decided by bank capacity. Same descriptors, reference volumes, split,
operators and seeds as the main configuration.
"""
import os, numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score

CACHE = "/content/drive/MyDrive"   # adjust if the caches live elsewhere
OUT   = "/content/drive/MyDrive/uad_results/predictions_patchcore.csv"
SEQS  = {"AXT1POST": "brain_cache_axt1post.npz",
         "AXFLAIR":  "brain_cache_axflair.npz",
         "AXT1":     "brain_cache_axt1.npz"}
FRACS, SEEDS = [0.01, 0.10], [0, 1, 2, 3, 4]
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, flush=True)


def coreset(X, m, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(len(X), (1,), generator=g))
    picked = [first]
    d = torch.cdist(X, X[first:first + 1]).squeeze(1)
    for _ in range(m - 1):
        j = int(torch.argmax(d))
        picked.append(j)
        d = torch.minimum(d, torch.cdist(X, X[j:j + 1]).squeeze(1))
    return torch.tensor(picked)


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
    qa  = np.concatenate([z["tn_dpatch"], z["ta_dpatch"]], 0)
    y   = np.r_[np.zeros(len(z["tn_dpatch"]), int), np.ones(len(z["ta_dpatch"]), int)]
    vol = np.r_[z["tn_vol"],   z["ta_vol"]]
    sl  = np.r_[z["tn_slice"], z["ta_slice"]]
    P = ref.shape[1]
    flat = torch.from_numpy(ref.reshape(-1, ref.shape[2])).to(dev)
    Q    = torch.from_numpy(qa.reshape(-1, qa.shape[2])).to(dev)
    print(seq, "pool", tuple(flat.shape), flush=True)

    for frac in FRACS:
        m = max(64, int(round(frac * len(flat))))
        for seed in SEEDS:
            bank = flat[coreset(flat, m, seed).to(dev)]
            pm = torch.cat([torch.cdist(Q[i:i + 8192], bank).min(1).values
                            for i in range(0, len(Q), 8192)]).cpu().numpy().reshape(len(qa), P)
            for op in ["max", "mean", "top1", "top5"]:
                s = agg(pm, op)
                rows += [dict(sequence=seq, level="patch",
                              backbone="patchcore_%d" % m, rule="coreset_knn",
                              operator=op, seed=seed, volume=str(v), slice=int(x),
                              y_true=int(t), score=float(c))
                         for v, x, t, c in zip(vol, sl, y, s)]
            print("   frac %.2f (m=%d) seed %d  max %.4f  top5 %.4f" %
                  (frac, m, seed, roc_auc_score(y, agg(pm, "max")),
                   roc_auc_score(y, agg(pm, "top5"))), flush=True)
    del flat, Q
    if dev == "cuda":
        torch.cuda.empty_cache()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print("wrote", OUT, len(rows), "rows")
