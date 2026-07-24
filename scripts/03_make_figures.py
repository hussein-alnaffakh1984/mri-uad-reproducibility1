#!/usr/bin/env python3
# Regenerates Figures 2-5 from the values in results/results_effects.csv
# and results/results_interaction.csv.
#   python scripts/03_make_figures.py --results results/ --out figures/
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

BLUE, ORANGE, RED = "#1F4E79", "#ED7D31", "#C00000"
SEQ = ["AXT1POST", "AXFLAIR", "AXT1"]
SUB = {"AXT1POST": "(contrast-enhanced)", "AXFLAIR": "(non-enhanced)", "AXT1": "(non-enhanced)"}

def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":15,"axes.labelsize":16,
      "xtick.labelsize":14,"ytick.labelsize":14,"legend.fontsize":14,"axes.linewidth":1.2,
      "xtick.major.width":1.2,"ytick.major.width":1.2,"xtick.major.size":6,
      "ytick.major.size":6,"axes.edgecolor":"#222222","figure.dpi":300,"savefig.dpi":300,
      "savefig.bbox":"tight","savefig.pad_inches":0.08})

def main(a):
    style(); os.makedirs(a.out, exist_ok=True)
    eff = pd.read_csv(os.path.join(a.results, "results_effects.csv"))
    itr = pd.read_csv(os.path.join(a.results, "results_interaction.csv"))
    opt = pd.read_csv(os.path.join(a.results, "results_opoint.csv"))
    val = lambda s, o, c: eff[(eff.sequence==s)&(eff.operator==o)][c].values[0]

    # Figure 2
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ops = ["max","top1","top5","mean"]
    nm  = {"max":"max","top1":"top-1%","top5":"top-5% (primary)","mean":"mean"}
    col = {"max":"#9DC3E6","top1":"#5B9BD5","top5":BLUE,"mean":ORANGE}
    x = np.arange(3); w = 0.19; lab = []
    for i, op in enumerate(ops):
        v = [val(s,op,"auroc_patch_seedmean") for s in SEQ]
        e = [val(s,op,"auroc_patch_seedsd")   for s in SEQ]
        xs = x + (i-1.5)*w
        ax.bar(xs, v, w, yerr=e, capsize=4, color=col[op], edgecolor="white",
               linewidth=0.9, label="patch: "+nm[op], error_kw={"elinewidth":1.6}, zorder=3)
        for xi, vi, ei, s in zip(xs, v, e, SEQ):
            lab.append((xi, vi+ei+0.008, vi, val(s,"top5","auroc_image")))
    for xi, s in zip(x, SEQ):
        b = val(s,"top5","auroc_image")
        ax.plot([xi-2.4*w, xi+2.4*w], [b,b], color=RED, lw=3.0, zorder=5)
    ax.plot([],[], color=RED, lw=3.0, label="image level (baseline, red line)")
    for xi, y, vi, b in lab:
        if y < b+0.014: y = b+0.016
        ax.text(xi, y, "%.3f"%vi, rotation=90, ha="center", va="bottom", fontsize=12.5,
                zorder=8, bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.92))
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n%s\nimage = %.3f"%(s, SUB[s], val(s,"top5","auroc_image")) for s in SEQ])
    for tk in ax.get_xticklabels(): tk.set_linespacing(1.5)
    ax.set_ylabel("AUROC"); ax.set_ylim(0.60, 0.96)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(axis="y", ls=":", alpha=0.45); ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5,1.01), ncol=3, frameon=False,
              handlelength=1.6, columnspacing=1.3)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    for ext in ("png","pdf"): fig.savefig(os.path.join(a.out, "fig2_operator_grid."+ext))
    plt.close(fig)

    # Figure 3
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    y = np.arange(3)[::-1]
    for yi, s in zip(y, SEQ):
        for op, c, off in [("top5",BLUE,0.17), ("mean",ORANGE,-0.17)]:
            v, lo, hi = val(s,op,"delta"), val(s,op,"ci_low"), val(s,op,"ci_high")
            ax.errorbar(v, yi+off, xerr=[[v-lo],[hi-v]], fmt="o", ms=11, color=c, ecolor=c,
                        elinewidth=2.4, capsize=6, capthick=2.4, zorder=4)
            ax.text(hi+0.008, yi+off, "%+.3f"%v, color=c, fontsize=13.5,
                    va="center", fontweight="bold")
    ax.axvline(0, color="#333333", lw=1.6, ls="--", zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(["%s\n%s"%(s, SUB[s]) for s in SEQ])
    ax.set_xlabel("\u0394 AUROC  (patch \u2212 image),  95% cluster-bootstrap CI")
    ax.plot([],[], "o", color=BLUE,   ms=11, label="locally concentrated (top-5%)")
    ax.plot([],[], "o", color=ORANGE, ms=11, label="full-slice averaging (mean)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5,1.01), ncol=2, frameon=False)
    ax.grid(axis="x", ls=":", alpha=0.45); ax.set_axisbelow(True)
    ax.set_xlim(-0.13, 0.27); ax.set_ylim(-0.55, 2.55)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    for ext in ("png","pdf"): fig.savefig(os.path.join(a.out, "fig3_operator_contrast."+ext))
    plt.close(fig)

    # Figure 4
    comp = itr[itr.comparison.astype(str).str.contains("-", na=False)].dropna(subset=["comparison"])
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    y = np.arange(len(comp))[::-1]
    for yi, r in zip(y, comp.itertuples()):
        sig = (r.ci_low > 0) or (r.ci_high < 0)
        c = BLUE if sig else "#7F7F7F"
        d = getattr(r, "diff", getattr(r, "estimate", None))
        ax.errorbar(d, yi, xerr=[[d-r.ci_low],[r.ci_high-d]], fmt="o", ms=13, color=c,
                    ecolor=c, elinewidth=2.6, capsize=7, capthick=2.6)
        ax.text(r.ci_high+0.008, yi, "%+.3f   p = %.3f%s"%(d, r.p, "  *" if sig else "  n.s."),
                fontsize=13.5, va="center", color=c, fontweight="bold" if sig else "normal")
    ax.axvline(0, color="#333333", lw=1.6, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([c.replace(" - "," \u2212 ") for c in comp.comparison])
    ax.set_xlabel("Difference in operator sensitivity  \u0394C,  95% CI")
    ax.grid(axis="x", ls=":", alpha=0.45); ax.set_axisbelow(True); ax.set_xlim(-0.08, 0.38)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    for ext in ("png","pdf"): fig.savefig(os.path.join(a.out, "fig4_interaction."+ext))
    plt.close(fig)

    # Figure 5
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(3); w = 0.34
    for ax_, metric, ttl, ylb in [
        (axes[0], "slice_sens", "Slice-level sensitivity\n(at 95th-percentile normal threshold)", "Sensitivity"),
        (axes[1], "volume_auroc_max", "Volume-level AUROC", "AUROC")]:
        for i, (lvl, c, lb) in enumerate([("image","#7F7F7F","image level"),
                                          ("patch",BLUE,"patch level (top-5%)")]):
            v = [opt[(opt.sequence==s)&(opt.level==lvl)][metric].values[0] for s in SEQ]
            bars = ax_.bar(x+(i-0.5)*w, v, w, color=c, edgecolor="white", linewidth=0.9, label=lb)
            ax_.bar_label(bars, fmt="%.3f", fontsize=13, padding=3)
        ax_.set_xticks(x); ax_.set_xticklabels(SEQ, fontsize=13.5)
        ax_.set_title(ttl, fontsize=15, pad=10); ax_.set_ylabel(ylb)
        ax_.set_ylim(0, 1.12); ax_.yaxis.set_major_locator(MultipleLocator(0.2))
        ax_.grid(axis="y", ls=":", alpha=0.45); ax_.set_axisbelow(True)
        for sp in ("top","right"): ax_.spines[sp].set_visible(False)
    axes[0].legend(loc="upper left", framealpha=0.96)
    for ext in ("png","pdf"): fig.savefig(os.path.join(a.out, "fig5_operating_points."+ext))
    plt.close(fig)
    print("figures written to", a.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/")
    ap.add_argument("--out", default="figures/")
    main(ap.parse_args())
