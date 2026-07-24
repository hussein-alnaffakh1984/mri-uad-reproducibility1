#!/usr/bin/env python3
# Builds the stored prediction file from the frozen feature caches.
# This is the only stage that needs a GPU-extracted cache; every table in the
# manuscript is regenerated from its output by scripts/02_analysis.py.
#
#   python scripts/01_build_predictions.py --caches /path/to/caches --out results/
#
# Expected cache per sequence (brain_cache_<seq>.npz), produced by the frozen
# encoders described in Section 3.5 of the manuscript:
#   ref_dino  (n,768)      ref_wrn  (n,1536)      ref_dpatch (n,256,768)
#   tn_dino / ta_dino      tn_wrn / ta_wrn        tn_dpatch / ta_dpatch
#   tn_vol / ta_vol        tn_slice / ta_slice
import argparse, json, os, time
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import roc_auc_score

K, BANK, SEEDS = 5, 20000, [0, 1, 2, 3, 4]
OPERATORS = ["max", "mean", "top1", "top5"]
SEQS = {"AXT1POST": "brain_cache_axt1post.npz",
        "AXFLAIR":  "brain_cache_axflair.npz",
        "AXT1":     "brain_cache_axt1.npz"}

def knn(ref, q, k=K):
    nn = NearestNeighbors(n_neighbors=k).fit(ref)
    return nn.kneighbors(q, n_neighbors=k)[0].mean(1)

def lof(ref, q, k=K):
    m = LocalOutlierFactor(n_neighbors=max(k, 20), novelty=True).fit(ref)
    return -m.score_samples(q)

def z(a):
    s = a.std()
    return (a - a.mean()) / (s if s > 0 else 1.0)

def patch_min_dist(bank, qp, chunk=16):
    out = np.empty(qp.shape[:2], np.float32); bn2 = (bank ** 2).sum(1)
    for i in range(0, len(qp), chunk):
        q = qp[i:i+chunk].reshape(-1, qp.shape[2]).astype(np.float32)
        d2 = (q ** 2).sum(1)[:, None] + bn2[None, :] - 2.0 * (q @ bank.T)
        out[i:i+chunk] = np.sqrt(np.maximum(d2, 0).min(1)).reshape(-1, qp.shape[1])
    return out

def aggregate(pm, op):
    if op == "max":  return pm.max(1)
    if op == "mean": return pm.mean(1)
    thr = np.quantile(pm, 0.99 if op == "top1" else 0.95, axis=1, keepdims=True)
    return np.array([p[p >= t].mean() for p, t in zip(pm, thr.ravel())])

def main(a):
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for seq, fn in SEQS.items():
        t0 = time.time()
        with np.load(os.path.join(a.caches, fn), allow_pickle=True) as zf:
            d = {k: zf[k] for k in zf.files}
        n_n, n_a = len(d["tn_dino"]), len(d["ta_dino"])
        y   = np.r_[np.zeros(n_n), np.ones(n_a)].astype(int)
        vol = np.r_[d["tn_vol"], d["ta_vol"]].astype(str)
        sl  = np.r_[d["tn_slice"], d["ta_slice"]].astype(int)

        def emit(level, backbone, rule, op, seed, score):
            rows.extend({"sequence": seq, "level": level, "backbone": backbone,
                         "rule": rule, "operator": op, "seed": seed, "volume": v,
                         "slice": int(s), "y_true": int(t), "score": float(x)}
                        for v, s, t, x in zip(vol, sl, y, score))

        img = {}
        for bb, kr, kn_, ka in [("dino","ref_dino","tn_dino","ta_dino"),
                                ("wrn","ref_wrn","tn_wrn","ta_wrn")]:
            R = d[kr]; Q = np.vstack([d[kn_], d[ka]])
            sk, sl_ = knn(R, Q), lof(R, Q)
            img[bb] = sk
            emit("image", bb, "knn",     "cls", -1, sk)
            emit("image", bb, "lof",     "cls", -1, sl_)
            emit("image", bb, "knn+lof", "cls", -1, z(sk) + z(sl_))
        emit("image", "ensemble", "knn", "cls", -1, z(img["dino"]) + z(img["wrn"]))

        refp = d["ref_dpatch"].reshape(-1, d["ref_dpatch"].shape[-1])
        qp   = np.vstack([d["tn_dpatch"], d["ta_dpatch"]])
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(refp), size=min(BANK, len(refp)), replace=False)
            pm  = patch_min_dist(np.ascontiguousarray(refp[idx].astype(np.float32)), qp)
            for op in OPERATORS:
                emit("patch", "dino", "knn", op, seed, aggregate(pm, op))
        print("%-9s %d normal / %d abnormal, %d volumes  (%.0fs)"
              % (seq, n_n, n_a, len(set(vol)), time.time() - t0))

    df = pd.DataFrame(rows)
    p = os.path.join(a.out, "predictions_brain.csv")
    df.to_csv(p, index=False)
    json.dump({"k": K, "bank": BANK, "seeds": SEEDS, "operators": OPERATORS,
               "n_rows": int(len(df)), "preprocessing": "raw (unnormalised) descriptors",
               "numpy": np.__version__},
              open(os.path.join(a.out, "predictions_brain_meta.json"), "w"), indent=2)
    print("wrote", p, len(df), "rows")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", required=True, help="directory holding brain_cache_*.npz")
    ap.add_argument("--out", default="results/")
    main(ap.parse_args())
