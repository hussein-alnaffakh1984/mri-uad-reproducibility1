"""AnomalyDINO comparator over all five seeds, Table 7, and the paired tests for
Tables 7 and 8 under the seed-averaged estimand.

L2-normalised patch tokens, cosine nearest neighbour to a patch memory bank of
20,000 vectors, slice score the mean of the top 1 per cent of patch distances.
Running it over all seeds is what allows Table 7 to use the same estimand as
Tables 3 and 4, so that every reported difference is the exact difference of the
two displayed columns.
"""
import os, numpy as np, pandas as pd
from scipy.stats import rankdata

CACHE, RES = "/content/drive/MyDrive", "/content/drive/MyDrive/uad_results"
URL = ("https://raw.githubusercontent.com/hussein-alnaffakh1984/"
       "mri-uad-reproducibility1/main/results/predictions_brain.csv")
SEQS = {"AXT1POST": "brain_cache_axt1post.npz",
        "AXFLAIR":  "brain_cache_axflair.npz",
        "AXT1":     "brain_cache_axt1.npz"}
BANK, SEEDS, B = 20000, [0, 1, 2, 3, 4], 2000
rng = np.random.default_rng(0)


def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def l2(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


def top_q(pm, q):
    thr = np.quantile(pm, q, axis=1, keepdims=True); m = pm >= thr
    return (pm * m).sum(1) / np.maximum(m.sum(1), 1)


rows = []
for seq, fn in SEQS.items():
    z   = np.load(os.path.join(CACHE, fn))
    ref = l2(z["ref_dpatch"].astype(np.float32))
    qa  = l2(np.concatenate([z["tn_dpatch"], z["ta_dpatch"]], 0).astype(np.float32))
    y   = np.r_[np.zeros(len(z["tn_dpatch"]), int), np.ones(len(z["ta_dpatch"]), int)]
    vol = np.r_[z["tn_vol"],   z["ta_vol"]]
    sl  = np.r_[z["tn_slice"], z["ta_slice"]]
    N, P, _ = qa.shape
    pool = ref.reshape(-1, ref.shape[2]); Q = qa.reshape(-1, qa.shape[2])
    for s in SEEDS:
        g = np.random.default_rng(s)
        bank = pool[np.sort(g.choice(len(pool), min(BANK, len(pool)), replace=False))]
        d = np.empty(len(Q), np.float32)
        for i in range(0, len(Q), 20000):
            d[i:i + 20000] = (1 - Q[i:i + 20000] @ bank.T).min(1)
        sc = top_q(d.reshape(N, P), 0.99)
        rows += [dict(sequence=seq, backbone="anomalydino", seed=s, volume=str(v),
                      slice=int(x), y_true=int(t), score=float(c))
                 for v, x, t, c in zip(vol, sl, y, sc)]
        print(seq, "seed", s, "AUROC %.4f" % auc(y, sc), flush=True)

ad = pd.DataFrame(rows)
ad.to_csv(f"{RES}/predictions_anomalydino.csv", index=False)

ours = pd.read_csv(URL)
dp = pd.read_csv(f"{RES}/predictions_dpmm_sens.csv")
dp = dp[dp.backbone == "dpmm_p128_t50"]


def mat(df, seq, seeds):
    out = []
    for s in seeds:
        g = df[(df.sequence == seq) & (df.seed == s)].copy()
        g["key"] = g.volume.astype(str) + "|" + g.slice.astype(str)
        out.append(g.sort_values("key"))
    return out


def holm(pv):
    o = np.argsort(pv); m = len(pv); out = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * pv[j]); out[j] = min(run, 1.0)
    return out


for name, df, seeds in [("AnomalyDINO", ad, SEEDS), ("DPMM", dp, [0, 1, 2])]:
    print(f"\n=== {name}: seed-averaged, paired volume bootstrap ===")
    ps = []
    for seq in ["AXT1POST", "AXFLAIR", "AXT1"]:
        A = mat(ours[(ours.level == "patch") & (ours.backbone == "dino") &
                     (ours.rule == "knn") & (ours.operator == "top5")], seq, SEEDS)
        Bm = mat(df, seq, seeds)
        k = sorted(set(A[0].key) & set(Bm[0].key))
        A = [g.set_index("key").loc[k] for g in A]
        Bm = [g.set_index("key").loc[k] for g in Bm]
        y = A[0].y_true.values.astype(int); vol = A[0].volume.astype(str).values
        uv = np.unique(vol); loc = {v: np.where(vol == v)[0] for v in uv}
        f = lambda M, i: np.mean([auc(y[i], g.score.values[i]) for g in M])
        allid = np.arange(len(y)); obs = f(A, allid) - f(Bm, allid); dist = []
        for _ in range(B):
            p = np.concatenate([loc[v] for v in rng.choice(uv, len(uv), True)])
            if y[p].min() == y[p].max():
                continue
            dist.append(f(A, p) - f(Bm, p))
        dist = np.array(dist)
        pv = max(2 * min((dist <= 0).mean(), (dist >= 0).mean()), 1 / len(dist))
        ps.append(pv)
        print("%-9s ours %.4f  %s %.4f  D %+.4f (%+.4f, %+.4f) p=%.4f"
              % (seq, f(A, allid), name, f(Bm, allid), obs,
                 np.percentile(dist, 2.5), np.percentile(dist, 97.5), pv))
    print("Holm:", np.round(holm(ps), 4))
