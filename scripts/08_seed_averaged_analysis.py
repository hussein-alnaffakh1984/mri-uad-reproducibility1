"""Seed-averaged estimand: Delta = mean_s AUROC(patch, seed s) - AUROC(image).
Because the image level is deterministic, this equals (five-seed mean) - (image),
so every reported difference is recoverable from the displayed columns (R1-7),
and inference is no longer restricted to a single seed (R2-5)."""
import numpy as np, pandas as pd
from scipy.stats import rankdata

d = pd.read_csv('results/predictions_brain.csv')
d = d[(d.backbone == 'dino') & (d.rule == 'knn')]
SEQ = ['AXT1POST', 'AXFLAIR', 'AXT1']
OPS = ['max', 'top1', 'top5', 'mean']
B, rng = 2000, np.random.default_rng(0)

def auc(y, s):
    r = rankdata(s)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

store = {}
for sq in SEQ:
    g = d[d.sequence == sq]
    im = g[g.level == 'image'].copy()
    im['key'] = im.volume.astype(str) + '|' + im.slice.astype(str)
    im = im.sort_values('key')
    y = im.y_true.values.astype(int)
    vol = im.volume.astype(str).values
    P = {}
    for op in OPS:
        M = []
        for s in range(5):
            p = g[(g.level == 'patch') & (g.operator == op) & (g.seed == s)].copy()
            p['key'] = p.volume.astype(str) + '|' + p.slice.astype(str)
            p = p.sort_values('key')
            assert (p.key.values == im.key.values).all()
            M.append(p.score.values)
        P[op] = np.array(M)
    store[sq] = dict(y=y, vol=vol, img=im.score.values, P=P)

def boot_idx(vol):
    uv = np.unique(vol); loc = {v: np.where(vol == v)[0] for v in uv}
    while True:
        pick = np.concatenate([loc[v] for v in rng.choice(uv, len(uv), True)])
        yield pick

def stat_delta(st, op, idx):
    y = st['y'][idx]
    if y.min() == y.max(): return None
    a_i = auc(y, st['img'][idx])
    return np.mean([auc(y, r[idx]) for r in st['P'][op]]) - a_i

def stat_C(st, idx):
    y = st['y'][idx]
    if y.min() == y.max(): return None
    return (np.mean([auc(y, r[idx]) for r in st['P']['top5']]) -
            np.mean([auc(y, r[idx]) for r in st['P']['mean']]))

def summarise(obs, dist):
    dist = np.array(dist)
    p = 2 * min((dist <= 0).mean(), (dist >= 0).mean())
    return obs, np.percentile(dist, 2.5), np.percentile(dist, 97.5), max(p, 1.0 / len(dist))

def holm(pv):
    o = np.argsort(pv); m = len(pv); out = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * pv[j]); out[j] = min(run, 1.0)
    return out

full = np.arange(len(store['AXT1POST']['y']))
print('=== TABLE 3 : seed-averaged Delta ===')
res3 = {}
for sq in SEQ:
    st = store[sq]
    n = len(st['y']); allidx = np.arange(n)
    a_img = auc(st['y'], st['img'])
    gen = boot_idx(st['vol'])
    idxs = [next(gen) for _ in range(B)]
    print('%s  image %.4f' % (sq, a_img))
    for op in OPS:
        obs = stat_delta(st, op, allidx)
        dist = [v for v in (stat_delta(st, op, i) for i in idxs) if v is not None]
        o, lo, hi, p = summarise(obs, dist)
        pm = np.mean([auc(st['y'], r) for r in st['P'][op]])
        res3[(sq, op)] = (a_img, pm, o, lo, hi, p)
        print('   %-5s patch %.4f  D %+.4f (%+.4f, %+.4f) p=%.4f   check %+.4f'
              % (op, pm, o, lo, hi, p, pm - a_img))

pv12 = [res3[(sq, op)][5] for sq in SEQ for op in OPS]
h12 = holm(pv12)
pv3 = [res3[(sq, 'top5')][5] for sq in SEQ]
h3 = holm(pv3)
print('\nHolm primary (3 top5):', np.round(h3, 4))
print('Holm full grid (12):  ', np.round(h12, 4))

print('\n=== TABLE 4 : operator contrast and interaction ===')
Cd = {}
for sq in SEQ:
    st = store[sq]; n = len(st['y'])
    gen = boot_idx(st['vol'])
    idxs = [next(gen) for _ in range(B)]
    obs = stat_C(st, np.arange(n))
    dist = [v for v in (stat_C(st, i) for i in idxs) if v is not None]
    Cd[sq] = np.array(dist)
    o, lo, hi, p = summarise(obs, dist)
    print('%-9s C %+.4f (%+.4f, %+.4f) p=%.4f' % (sq, o, lo, hi, p))

pairs = [('AXT1POST', 'AXFLAIR'), ('AXT1POST', 'AXT1'), ('AXFLAIR', 'AXT1')]
pi = []
print()
for a, b in pairs:
    m = min(len(Cd[a]), len(Cd[b]))
    dist = Cd[a][:m] - Cd[b][:m]
    obs = (stat_C(store[a], np.arange(len(store[a]['y']))) -
           stat_C(store[b], np.arange(len(store[b]['y']))))
    o, lo, hi, p = summarise(obs, dist)
    pi.append(p)
    print('%-9s - %-9s dC %+.4f (%+.4f, %+.4f) p=%.4f' % (a, b, o, lo, hi, p))
print('Holm interaction:', np.round(holm(pi), 4))
