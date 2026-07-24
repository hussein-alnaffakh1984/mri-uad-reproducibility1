#!/usr/bin/env python3
# Regenerates every table in the manuscript from the stored prediction file.
# No feature extraction required.
#   python scripts/02_analysis.py --results results/ --out results/regenerated/
import argparse, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

B, SEED, PRIMARY = 2000, 0, 'top5'
SEQ = ['AXT1POST', 'AXFLAIR', 'AXT1']

def auc(y, s):
    return roc_auc_score(y, s) if len(set(y)) > 1 else np.nan

def paired_boot(d_a, d_b, B=B, seed=SEED):
    m = d_a[['volume','slice','y_true','score']].merge(
        d_b[['volume','slice','score']], on=['volume','slice'], suffixes=('_a','_b'))
    vols = m.volume.unique(); rng = np.random.default_rng(seed)
    obs = auc(m.y_true, m.score_a) - auc(m.y_true, m.score_b)
    idx = {v: m[m.volume == v] for v in vols}; out = []
    for _ in range(B):
        s = pd.concat([idx[v] for v in rng.choice(vols, len(vols), True)], ignore_index=True)
        if s.y_true.nunique() > 1:
            out.append(auc(s.y_true, s.score_a) - auc(s.y_true, s.score_b))
    out = np.array(out)
    p = max(2 * min((out <= 0).mean(), (out >= 0).mean()), 1 / B)
    return obs, np.percentile(out, 2.5), np.percentile(out, 97.5), p, out

def holm(pv):
    pv = np.asarray(pv, dtype=float); order = np.argsort(pv)
    adj = np.empty_like(pv); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(pv) - r) * pv[i]); adj[i] = min(run, 1.0)
    return adj

def main(a):
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(os.path.join(a.results, 'predictions_brain.csv'))
    def get(sq, lvl, op, sd):
        return df[(df.sequence == sq) & (df.level == lvl) & (df.operator == op) &
                  (df.seed == sd) & (df.backbone == 'dino') & (df.rule == 'knn')]

    rows = []
    for sq in SEQ:
        base = get(sq, 'image', 'cls', -1)
        a_img = auc(base.y_true, base.score)
        for op in ['max', 'top1', 'top5', 'mean']:
            per_seed = []
            for sd in sorted(df.seed.unique()):
                if sd < 0: continue
                g = get(sq, 'patch', op, sd)
                if len(g): per_seed.append(auc(g.y_true, g.score))
            obs, lo, hi, p, _ = paired_boot(get(sq, 'patch', op, SEED), base)
            rows.append(dict(sequence=sq, image=round(a_img, 4), operator=op,
                             patch_mean=round(float(np.mean(per_seed)), 4),
                             patch_sd=round(float(np.std(per_seed)), 4),
                             delta=round(obs, 4), ci_low=round(lo, 4),
                             ci_high=round(hi, 4), p_unadj=round(p, 4)))
    t3 = pd.DataFrame(rows)
    m = t3.operator == PRIMARY
    t3['p_holm_primary'] = np.nan
    t3.loc[m, 'p_holm_primary'] = holm(t3.loc[m, 'p_unadj'].values)
    t3['p_holm_full'] = holm(t3['p_unadj'].values)
    t3.to_csv(os.path.join(a.out, 'table3_operator_grid.csv'), index=False)

    dist, rows = {}, []
    for sq in SEQ:
        obs, lo, hi, p, d = paired_boot(get(sq, 'patch', 'top5', SEED),
                                        get(sq, 'patch', 'mean', SEED))
        dist[sq] = d
        rows.append(dict(comparison=sq, kind='within', estimate=round(obs, 4),
                         ci_low=round(lo, 4), ci_high=round(hi, 4), p=round(p, 4)))
    inter = []
    for x, y in [('AXT1POST','AXFLAIR'), ('AXT1POST','AXT1'), ('AXFLAIR','AXT1')]:
        n = min(len(dist[x]), len(dist[y])); d = dist[x][:n] - dist[y][:n]
        p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / B)
        inter.append(dict(comparison=x + ' - ' + y, kind='between',
                          estimate=round(float(d.mean()), 4),
                          ci_low=round(float(np.percentile(d, 2.5)), 4),
                          ci_high=round(float(np.percentile(d, 97.5)), 4), p=round(p, 4)))
    for r, ph in zip(inter, holm([r['p'] for r in inter])):
        r['p_holm'] = round(float(ph), 4)
    pd.DataFrame(rows + inter).to_csv(os.path.join(a.out, 'table4_operator_contrast.csv'), index=False)

    rows = []
    for sq in SEQ:
        for lvl, op, sd, lab in [('image','cls',-1,'image'), ('patch',PRIMARY,SEED,'patch')]:
            d = get(sq, lvl, op, sd)
            if not len(d): continue
            vs = d.groupby('volume').agg(y=('y_true','max'), s=('score','max'))
            for tgt in (0.90, 0.95, 0.99):
                thr = np.quantile(d[d.y_true == 0].score, tgt)
                g = d.groupby('volume').agg(y=('y_true','max'),
                                            fp=('score', lambda s: (s > thr).sum()),
                                            hit=('score', lambda s: (s > thr).any()))
                rows.append(dict(sequence=sq, level=lab,
                                 volume_auroc=round(auc(vs.y, vs.s), 4), spec_target=tgt,
                                 slice_sens=round(float((d[d.y_true == 1].score > thr).mean()), 4),
                                 volume_sens=round(float(g[g.y == 1].hit.mean()), 4),
                                 fp_per_normal_volume=round(float(g[g.y == 0].fp.mean()), 3)))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, 'table5_operating_points.csv'), index=False)

    rows = []
    for sq in SEQ:
        for bb in ['dino', 'wrn', 'ensemble']:
            d = df[(df.sequence == sq) & (df.level == 'image') &
                   (df.backbone == bb) & (df.rule == 'knn')]
            if not len(d): continue
            vols = d.volume.unique(); rng = np.random.default_rng(SEED)
            idx = {v: d[d.volume == v] for v in vols}; bt = []
            for _ in range(B):
                s = pd.concat([idx[v] for v in rng.choice(vols, len(vols), True)], ignore_index=True)
                if s.y_true.nunique() > 1: bt.append(auc(s.y_true, s.score))
            rows.append(dict(sequence=sq, backbone=bb, auroc=round(auc(d.y_true, d.score), 4),
                             ci_low=round(float(np.percentile(bt, 2.5)), 4),
                             ci_high=round(float(np.percentile(bt, 97.5)), 4)))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, 'table6_backbone.csv'), index=False)
    print('regenerated tables ->', a.out)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/')
    ap.add_argument('--out', default='results/regenerated/')
    main(ap.parse_args())
