"""
generate_new_figures.py
-----------------------
Generates 4 new figures (+ a printed analysis block) for the EDA chapter
of the TFG on Engagement Prediction of Short Videos.

Run from the repo root:
    conda run -n tfg-ugc-retention python scripts/generate_new_figures.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths and global style
# ---------------------------------------------------------------------------
BASE_RAW  = 'data/raw/'
BASE_PROC = 'data/processed/'

plt.rcParams.update({
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': '#F9F9F9',
    'figure.facecolor': 'white',
})

BLUE  = '#2E6FA3'
CORAL = '#D95F4B'
GREEN = '#4A9B7F'

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
train = pd.read_csv(BASE_RAW + 'train_data.csv')
meta  = pd.read_csv(BASE_PROC + 'train_metadata.csv')
df = train.merge(meta, on='Id', how='inner')
df['has_title']       = df['Title'].notna() & (df['Title'].str.strip() != '')
df['has_description'] = df['Description'].notna() & (df['Description'].str.strip() != '')
df['resolution']      = df['width'] * df['height']

df_short = df[df['duration'] < 10]
df_long  = df[df['duration'] >= 10]
df_clean = df[(df['duration'] >= 5) & (df['duration'] <= 60)]

# ---------------------------------------------------------------------------
# FIGURE 1 — Bimodal side-by-side
# ---------------------------------------------------------------------------

def plot_ecr_with_kde(ax, data, color, title, n_label, show_annotations=False):
    """Helper: histogram + KDE on a single axes."""
    ecr = data['ECR'].dropna()
    counts, bins, _ = ax.hist(ecr, bins=50, edgecolor='white', linewidth=0.4,
                               alpha=0.70, color=color)
    bin_width = bins[1] - bins[0]
    kde = gaussian_kde(ecr, bw_method=0.10)
    xs  = np.linspace(0, 1, 1000)
    kde_scaled = kde(xs) * len(ecr) * bin_width
    ax.plot(xs, kde_scaled, color=color, lw=2.0, alpha=0.9)

    mean_ecr = ecr.mean()
    ax.axvline(mean_ecr, color=color, lw=1.5, ls='--', alpha=0.85)
    # Use a fixed y-reference so the label does not depend on get_ylim() before
    # the axes limits are finalised.
    ymax = ax.get_ylim()[1]
    ax.text(mean_ecr + 0.02, ymax * 0.90,
            f'\u03bc = {mean_ecr:.3f}', color=color, fontsize=9.5)

    if show_annotations:
        yref = ax.get_ylim()[1] * 0.5
        # Peak 1: swipe-away zone (~0.1)
        ax.annotate('Pico "swipe"\n(ECR \u2248 0.1)',
                    xy=(0.10, yref),
                    xytext=(0.24, yref * 0.7),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0),
                    fontsize=8.5, color='#555555', ha='center')
        # Peak 2: retention zone (~0.9)
        ax.annotate('Pico "retenci\u00f3n"\n(ECR \u2248 0.9)',
                    xy=(0.88, yref),
                    xytext=(0.72, yref * 0.7),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0),
                    fontsize=8.5, color='#555555', ha='center')

    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('ECR \u2014 ranking normalizado de retenci\u00f3n', fontsize=10)
    ax.set_ylabel('N\u00famero de v\u00eddeos', fontsize=10)
    ax.text(0.98, 0.94, n_label, transform=ax.transAxes,
            ha='right', va='top', fontsize=9, color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc'))
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='grey')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.subplots_adjust(bottom=0.20, wspace=0.35)

plot_ecr_with_kde(ax1, df, BLUE,
                  '(a) Dataset completo',
                  f'N = {len(df):,}',
                  show_annotations=False)

df_ge10 = df[df['duration'] >= 10]
plot_ecr_with_kde(ax2, df_ge10, BLUE,
                  '(b) V\u00eddeos \u2265 10 s',
                  f'N = {len(df_ge10):,}',
                  show_annotations=True)

fig.suptitle('Distribuci\u00f3n del ECR seg\u00fan umbral de duraci\u00f3n',
             fontsize=14, fontweight='bold', y=1.01)

fig.text(0.5, 0.04,
    'Comparativa de la distribuci\u00f3n del ECR para el dataset completo (a) y filtrado a v\u00eddeos de duraci\u00f3n \u2265 10 s (b).\n'
    'Los v\u00eddeos de 5\u20139 s generan valores ECR distribuidos uniformemente (ruido de reacci\u00f3n), enmascarando el patr\u00f3n bimodal.',
    ha='center', va='bottom', fontsize=8.5, color='#444444', style='italic')

plt.savefig('results/ecr_bimodal_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\u2713 Figure 1 saved: results/ecr_bimodal_comparison.png")

# ---------------------------------------------------------------------------
# FIGURE 2 — ECR por bucket de duracion
# ---------------------------------------------------------------------------
df_b = df_clean.copy()
bins_dur   = [5, 10, 30, 60]
labels_dur = ['5\u201310 s\n(corto)', '10\u201330 s\n(medio)', '30\u201360 s\n(largo)']
df_b['duration_bucket'] = pd.cut(df_b['duration'], bins=bins_dur, labels=labels_dur, right=True)
df_b = df_b.dropna(subset=['duration_bucket', 'ECR'])

bucket_colors = [CORAL, BLUE, GREEN]
bucket_counts = df_b.groupby('duration_bucket', observed=True).size()
bucket_means  = df_b.groupby('duration_bucket', observed=True)['ECR'].mean()

fig, ax = plt.subplots(figsize=(9, 6))
fig.subplots_adjust(bottom=0.20)

positions = [1, 2, 3]
vp = ax.violinplot(
    [df_b[df_b['duration_bucket'] == lbl]['ECR'].values for lbl in labels_dur],
    positions=positions, widths=0.6, showmedians=False, showextrema=False
)
for i, pc in enumerate(vp['bodies']):
    pc.set_facecolor(bucket_colors[i])
    pc.set_alpha(0.35)
    pc.set_edgecolor(bucket_colors[i])

bp = ax.boxplot(
    [df_b[df_b['duration_bucket'] == lbl]['ECR'].values for lbl in labels_dur],
    positions=positions, widths=0.25, patch_artist=True,
    medianprops=dict(color='white', lw=2.5),
    whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
    flierprops=dict(marker='o', markersize=1.5, alpha=0.2)
)
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(bucket_colors[i])
    patch.set_alpha(0.85)
    patch.set_edgecolor(bucket_colors[i])
for element in ['whiskers', 'caps', 'fliers']:
    for i, item in enumerate(bp[element]):
        item.set_color(bucket_colors[i // 2])

# Mean diamonds
for i, (pos, mean_val) in enumerate(zip(positions, bucket_means)):
    ax.scatter(pos, mean_val, marker='D', s=60, color='white',
               zorder=5, edgecolors=bucket_colors[i], lw=1.5)
    ax.text(pos, mean_val + 0.035, f'\u03bc={mean_val:.3f}',
            ha='center', fontsize=8.5, color=bucket_colors[i], fontweight='bold')

# N annotations on x-axis
xlabels_with_n = [f'{lbl}\n(N={bucket_counts[lbl]:,})' for lbl in labels_dur]
ax.set_xticks(positions)
ax.set_xticklabels(xlabels_with_n, fontsize=10)
ax.set_ylabel('ECR \u2014 ranking normalizado de retenci\u00f3n', fontsize=11)
ax.set_title('Distribuci\u00f3n del ECR por intervalo de duraci\u00f3n\n'
             '(proxy de categor\u00eda de contenido)',
             fontsize=12, fontweight='bold', pad=10)
ax.axhline(df_b['ECR'].mean(), color='#888888', lw=1.0, ls=':', alpha=0.7)
ax.text(3.55, df_b['ECR'].mean() + 0.01, f'Media global: {df_b["ECR"].mean():.3f}',
        fontsize=8.5, color='#888888')
ax.grid(axis='y', linestyle='--', alpha=0.35, color='grey')
ax.set_ylim(-0.05, 1.12)

fig.text(0.5, 0.04,
    'Distribuci\u00f3n del ECR por intervalo de duraci\u00f3n (5\u201360 s). El dataset SnapUGC v2 no contiene etiquetas\n'
    'de categor\u00eda; los intervalos de duraci\u00f3n se emplean como variable proxy. Diamantes = media de ECR.',
    ha='center', va='bottom', fontsize=8.5, color='#444444', style='italic')

plt.savefig('results/ecr_duration_buckets.png', dpi=300, bbox_inches='tight')
plt.close()
print("\u2713 Figure 2 saved: results/ecr_duration_buckets.png")

# ---------------------------------------------------------------------------
# FIGURE 3 — ECR por metadatos textuales
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.subplots_adjust(bottom=0.20, wspace=0.35)

for ax, col, col_label in [(ax1, 'has_title', 'T\u00edtulo'), (ax2, 'has_description', 'Descripci\u00f3n')]:
    groups    = [False, True]
    labels_g  = ['Sin ' + col_label, 'Con ' + col_label]
    colors_g  = [CORAL, BLUE]
    data_g    = [df[df[col] == g]['ECR'].dropna().values for g in groups]
    counts_g  = [len(d) for d in data_g]
    means_g   = [d.mean() for d in data_g]

    vp = ax.violinplot(data_g, positions=[1, 2], widths=0.55,
                       showmedians=False, showextrema=False)
    for i, pc in enumerate(vp['bodies']):
        pc.set_facecolor(colors_g[i])
        pc.set_alpha(0.4)

    bp = ax.boxplot(data_g, positions=[1, 2], widths=0.22, patch_artist=True,
                    medianprops=dict(color='white', lw=2.5),
                    whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors_g[i])
        patch.set_alpha(0.85)
        patch.set_edgecolor(colors_g[i])
    for el in ['whiskers', 'caps']:
        for i, item in enumerate(bp[el]):
            item.set_color(colors_g[i // 2])

    for i, (pos, mean_val) in enumerate(zip([1, 2], means_g)):
        ax.scatter(pos, mean_val, marker='D', s=55, color='white',
                   zorder=5, edgecolors=colors_g[i], lw=1.5)
        ax.text(pos, mean_val + 0.04, f'\u03bc={mean_val:.3f}',
                ha='center', fontsize=9, color=colors_g[i], fontweight='bold')

    xlabels_n = [f'{lbl}\n(N={c:,})' for lbl, c in zip(labels_g, counts_g)]
    ax.set_xticks([1, 2])
    ax.set_xticklabels(xlabels_n, fontsize=10)
    ax.set_ylabel('ECR \u2014 ranking normalizado de retenci\u00f3n', fontsize=10)
    ax.set_title(f'ECR seg\u00fan presencia de {col_label.lower()}',
                 fontsize=11, fontweight='bold', pad=8)
    ax.set_ylim(-0.05, 1.12)
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='grey')

fig.suptitle('ECR por disponibilidad de metadatos textuales',
             fontsize=13, fontweight='bold', y=1.01)
fig.text(0.5, 0.04,
    'Distribuci\u00f3n del ECR (ranking normalizado) seg\u00fan presencia de t\u00edtulo y descripci\u00f3n.\n'
    'Dataset SnapUGC train (N=106,192). Diamantes = media del grupo.',
    ha='center', va='bottom', fontsize=8.5, color='#444444', style='italic')

plt.savefig('results/ecr_text_metadata.png', dpi=300, bbox_inches='tight')
plt.close()
print("\u2713 Figure 3 saved: results/ecr_text_metadata.png")

# ---------------------------------------------------------------------------
# ANALYSIS CELL — printed output, no figure
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("AN\u00c1LISIS DE ANOMAL\u00cdAS Y CASOS EXTREMOS")
print("=" * 55)

n_ecr_zero  = (df['ECR'] == 0.0).sum()
n_ecr_one   = (df['ECR'] >= 0.999).sum()
n_long      = (df['duration'] > 61).sum()
n_no_audio  = (df['has_audio'] == 0).sum() if 'has_audio' in df.columns else "N/A"

print(f"ECR = 0.0 exactamente:    {n_ecr_zero:,} v\u00eddeos ({n_ecr_zero/len(df)*100:.2f}%)")
print(f"ECR \u2265 0.999 (techo):      {n_ecr_one:,} v\u00eddeos ({n_ecr_one/len(df)*100:.2f}%)")
print(f"Duraci\u00f3n > 61s (viola spec): {n_long:,} v\u00eddeos")
print(f"Sin audio (has_audio=0):   {n_no_audio}")
print()
print("Decisi\u00f3n de limpieza: MANTENER todos (no filtrar).")
print("  - ECR=0.0 son casos reales de abandono inmediato.")
print("  - Los 21 v\u00eddeos >61s son outliers de medici\u00f3n; representan <0.02%.")
print("  - 99.9% tienen audio \u2192 variable near-constant, no informativa como predictor.")
print("  - Se documenta en sec. 4.5 (Partici\u00f3n, limpieza y preparaci\u00f3n).")
print("=" * 55)
