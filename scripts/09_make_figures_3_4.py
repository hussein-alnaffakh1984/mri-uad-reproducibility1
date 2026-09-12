"""Regenerate Figures 3 and 4 under the seed-averaged estimand.
Colours and layout follow the originals: dark blue 1F4E79, orange ED7D31."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, GREY = '#1F4E79', '#ED7D31', '#7F7F7F'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 20,
                     'axes.edgecolor': '#222222', 'axes.linewidth': 1.4})

SEQ = ['AXT1POST\n(contrast-enhanced)', 'AXFLAIR\n(non-enhanced)', 'AXT1\n(non-enhanced)']
TOP5 = [(+0.1006, 0.0561, 0.1448), (+0.0396, 0.0091, 0.0717), (+0.0548, 0.0104, 0.0983)]
MEAN = [(-0.0335, -0.0795, 0.0150), (+0.0290, -0.0099, 0.0693), (+0.0561, 0.0039, 0.1154)]

fig, ax = plt.subplots(figsize=(15.6, 8.3), dpi=200)
for i, (t, m) in enumerate(zip(TOP5, MEAN)):
    yb, yo = 2 - i + 0.16, 2 - i - 0.16
    ax.errorbar(t[0], yb, xerr=[[t[0]-t[1]], [t[2]-t[0]]], fmt='o', ms=13,
                color=BLUE, ecolor=BLUE, elinewidth=2.6, capsize=8, capthick=2.6)
    ax.errorbar(m[0], yo, xerr=[[m[0]-m[1]], [m[2]-m[0]]], fmt='o', ms=13,
                color=ORANGE, ecolor=ORANGE, elinewidth=2.6, capsize=8, capthick=2.6)
    ax.text(t[2]+0.008, yb, '%+.3f' % t[0], va='center', color=BLUE, fontweight='bold')
    ax.text(m[2]+0.008, yo, '%+.3f' % m[0], va='center', color=ORANGE, fontweight='bold')
ax.axvline(0, ls='--', color='#444444', lw=1.6)
ax.set_yticks([2, 1, 0]); ax.set_yticklabels(SEQ)
ax.set_xlim(-0.12, 0.24)
ax.set_xlabel('\u0394 AUROC  (patch \u2212 image),  95% cluster-bootstrap CI', labelpad=12)
ax.spines[['top', 'right']].set_visible(False)
h = [plt.Line2D([], [], marker='o', ls='', color=BLUE, ms=13,
                label='locally concentrated (top-5%)'),
     plt.Line2D([], [], marker='o', ls='', color=ORANGE, ms=13,
                label='full-slice averaging (mean)')]
ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=2,
          frameon=False)
fig.tight_layout(); fig.savefig('figures/figure3.png', dpi=200,
                                bbox_inches='tight', facecolor='white')
print('Figure 3 rewritten', end=' ')
from PIL import Image; print(Image.open('figures/figure3.png').size)

PAIR = ['AXT1POST \u2212 AXFLAIR', 'AXT1POST \u2212 AXT1', 'AXFLAIR \u2212 AXT1']
DC = [(+0.1235, 0.0333, 0.2019, 0.045, True),
      (+0.1354, 0.0375, 0.2177, 0.045, True),
      (+0.0119, -0.0485, 0.0737, 0.735, False)]
fig, ax = plt.subplots(figsize=(14.6, 6.2), dpi=200)
for i, (v, lo, hi, p, sig) in enumerate(DC):
    y = 2 - i
    c = BLUE if sig else GREY
    ax.errorbar(v, y, xerr=[[v-lo], [hi-v]], fmt='o', ms=14, color=c, ecolor=c,
                elinewidth=2.8, capsize=9, capthick=2.8)
    lab = '%+.3f   Holm p = %.3f %s' % (v, p, '*' if sig else 'n.s.')
    ax.text(hi+0.008, y, lab, va='center', color=c, fontweight='bold')
ax.axvline(0, ls='--', color='#444444', lw=1.6)
ax.set_yticks([2, 1, 0]); ax.set_yticklabels(PAIR)
ax.set_xlim(-0.08, 0.42)
ax.set_xlabel('Difference in operator sensitivity  \u0394C,  95% CI', labelpad=12)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig('figures/figure4.png', dpi=200,
                                bbox_inches='tight', facecolor='white')
print('Figure 4 rewritten', Image.open('figures/figure4.png').size)
