#!/usr/bin/env python3
"""Generate two structured EDA notebooks from existing notebook content."""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

NB_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')

# ═══════════════════════════════════════════════════════════════
# NOTEBOOK 1: EstadísticasDescriptivasyDistribuciones.ipynb
# ═══════════════════════════════════════════════════════════════

cells_edd = []

cells_edd.append(new_markdown_cell(
    "# EDA — Estadísticas Descriptivas y Distribuciones\n"
    "## Dataset SnapUGC v2 — ICCV VQualA Challenge 2025\n\n"
    "Este notebook cubre el análisis exploratorio de las distribuciones del ECR y la duración\n"
    "en el dataset SnapUGC v2, tanto a nivel del conjunto de entrenamiento como en comparación\n"
    "entre las tres particiones oficiales (train / val / test).\n\n"
    "**Aporte original OE1**: Identificación y caracterización de sesgos sistemáticos en la\n"
    "distribución del ECR entre los subconjuntos del challenge, no documentados en los trabajos\n"
    "de referencia.\n\n"
    "Dataset: 106,192 (train) + 6,000 (val) + 8,459 (test) = 120,651 vídeos totales."
))

cells_edd.append(new_code_cell(
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.ticker as ticker\n"
    "from scipy import stats\n"
    "from scipy.stats import gaussian_kde\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n\n"
    "plt.rcParams.update({\n"
    "    'font.family': 'serif',\n"
    "    'axes.spines.top': False,\n"
    "    'axes.spines.right': False,\n"
    "    'axes.facecolor': '#F9F9F9',\n"
    "    'figure.facecolor': 'white',\n"
    "})\n\n"
    "BASE_RAW  = '../data/raw/'\n"
    "BASE_PROC = '../data/processed/'\n"
    "BLUE   = '#2E6FA3'\n"
    "CORAL  = '#D95F4B'\n"
    "GREEN  = '#4A9B7F'"
))

cells_edd.append(new_code_cell(
    "# ── TRAIN\n"
    "train_data = pd.read_csv(BASE_RAW + 'train_data.csv')\n"
    "train_meta = pd.read_csv(BASE_PROC + 'train_metadata.csv')\n"
    "df_train   = train_data.merge(train_meta, on='Id', how='inner')\n"
    "df_train['has_title']       = df_train['Title'].notna() & (df_train['Title'].str.strip() != '')\n"
    "df_train['has_description'] = df_train['Description'].notna() & (df_train['Description'].str.strip() != '')\n"
    "df_train['split'] = 'train'\n\n"
    "# Subconjuntos por duración (train)\n"
    "df_10s_to_60s = df_train[(df_train['duration'] >= 10) & (df_train['duration'] <= 60)]\n\n"
    "# ── VALIDATION\n"
    "val_data = pd.read_csv(BASE_RAW + 'val_data.csv')\n"
    "val_meta = pd.read_csv(BASE_PROC + 'val_metadata.csv')\n"
    "df_val   = val_data.merge(val_meta, on='Id', how='inner')\n"
    "df_val['has_title']       = df_val['Title'].notna() & (df_val['Title'].str.strip() != '')\n"
    "df_val['has_description'] = df_val['Description'].notna() & (df_val['Description'].str.strip() != '')\n"
    "df_val['split'] = 'val'\n\n"
    "# ── TEST\n"
    "test_data = pd.read_csv(BASE_RAW + 'test_data.csv')\n"
    "test_meta = pd.read_csv(BASE_PROC + 'test_metadata.csv')\n"
    "df_test   = test_data.merge(test_meta, on='Id', how='inner')\n"
    "df_test['has_title']       = df_test['Title'].notna() & (df_test['Title'].str.strip() != '')\n"
    "df_test['has_description'] = df_test['Description'].notna() & (df_test['Description'].str.strip() != '')\n"
    "df_test['split'] = 'test'\n\n"
    "print(f'Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}')\n"
    "print(f'Total: {len(df_train)+len(df_val)+len(df_test):,}')\n"
    "print(f'\\nTrain — vídeos <10s:   {(df_train[\"duration\"] < 10).sum():,} ({(df_train[\"duration\"] < 10).mean()*100:.1f}%)')\n"
    "print(f'Train — vídeos 10-60s: {len(df_10s_to_60s):,} ({len(df_10s_to_60s)/len(df_train)*100:.1f}%)')\n"
    "print(f'Train — vídeos >60s:   {(df_train[\"duration\"] > 60).sum():,}')"
))

cells_edd.append(new_markdown_cell("## 1. Estadísticas descriptivas por partición"))

cells_edd.append(new_code_cell(
    "pcts = [5, 25, 50, 75, 95]\n"
    "rows = []\n"
    "for name, df in [('train', df_train), ('val', df_val), ('test', df_test)]:\n"
    "    ecr = df['ECR'].dropna()\n"
    "    dur = df['duration'].dropna()\n"
    "    row = {\n"
    "        'Split': name,\n"
    "        'N': f'{len(df):,}',\n"
    "        'ECR mean': f'{ecr.mean():.3f}',\n"
    "        'ECR std':  f'{ecr.std():.3f}',\n"
    "        **{f'ECR P{p}': f'{np.percentile(ecr, p):.3f}' for p in pcts},\n"
    "        'Dur mean (s)': f'{dur.mean():.1f}',\n"
    "        'Dur median (s)': f'{np.median(dur):.1f}',\n"
    "        'Dur std (s)': f'{dur.std():.1f}',\n"
    "        **{f'Dur P{p}': f'{np.percentile(dur, p):.1f}' for p in pcts},\n"
    "        'has_title (%)': f'{df[\"has_title\"].mean()*100:.1f}',\n"
    "        'has_desc (%)':  f'{df[\"has_description\"].mean()*100:.1f}',\n"
    "        '% < 10s': f'{(df[\"duration\"] < 10).mean()*100:.1f}',\n"
    "    }\n"
    "    rows.append(row)\n\n"
    "stats_df = pd.DataFrame(rows).set_index('Split')\n"
    "print(stats_df.T.to_string())"
))

cells_edd.append(new_markdown_cell(
    "### Interpretación: Estadísticas descriptivas\n\n"
    "Las tres particiones son estadísticamente muy similares en todas las variables analizadas.\n"
    "El ECR presenta una distribución casi uniforme en [0, 1] con media ~0.498 y desviación\n"
    "típica ~0.290 en las tres particiones. La duración tiene una distribución log-normal\n"
    "con mediana ~9.7 s, reflejo del formato dominante de vídeo muy corto en la plataforma.\n\n"
    "Destaca que aproximadamente el **52% de los vídeos en cada partición tiene duración\n"
    "inferior a 10 segundos** — una propiedad estructural del dataset v2, no un artefacto\n"
    "del muestreo (véase Sección 3)."
))

cells_edd.append(new_markdown_cell(
    "## 2. Distribución del ECR — Conjunto de entrenamiento\n\n"
    "### 2.1 Dataset completo (5–60 s)"
))

cells_edd.append(new_code_cell(
    "fig, ax = plt.subplots(figsize=(7, 5))\n"
    "fig.subplots_adjust(bottom=0.22)\n\n"
    "ecr_data = df_train['ECR'].dropna()\n"
    "counts, bins, _ = ax.hist(ecr_data, bins=50, edgecolor='white', linewidth=0.4, alpha=0.75, color=BLUE)\n\n"
    "kde = gaussian_kde(ecr_data, bw_method=0.10)\n"
    "xs = np.linspace(ecr_data.min(), ecr_data.max(), 1200)\n"
    "bin_width = bins[1] - bins[0]\n"
    "kde_scaled = kde(xs) * len(ecr_data) * bin_width\n"
    "ax.plot(xs, kde_scaled, color=BLUE, lw=2, alpha=0.85)\n\n"
    "mean_ecr = ecr_data.mean()\n"
    "ax.axvline(mean_ecr, color=BLUE, lw=1.5, ls='--', alpha=0.9)\n"
    "ax.text(mean_ecr + 0.01, ax.get_ylim()[1] * 0.92, f'μ = {mean_ecr:.3f}', color=BLUE, fontsize=10)\n\n"
    "ax.set_xlabel('ECR — ranking normalizado de retención', fontsize=11)\n"
    "ax.set_ylabel('Número de vídeos', fontsize=11)\n"
    "ax.set_title('Distribución del ECR — Dataset completo (5–60 s)', fontsize=13, fontweight='bold', pad=10)\n"
    "ax.grid(axis='y', linestyle='--', alpha=0.35, color='grey')\n\n"
    "fig.text(0.5, 0.04,\n"
    "    f'Distribución del ECR en el dataset SnapUGC v2 (N = {len(df_train):,} vídeos de entrenamiento). '\n"
    "    'La línea discontinua indica la media muestral.',\n"
    "    ha='center', va='top', fontsize=9, color='#444444', style='italic')\n\n"
    "plt.savefig('../results/ecr_dist.png', dpi=300, bbox_inches='tight')\n"
    "plt.show()"
))

cells_edd.append(new_markdown_cell("### 2.2 Filtrado: vídeos ≥ 10 s"))

cells_edd.append(new_code_cell(
    "fig, ax = plt.subplots(figsize=(7, 5))\n"
    "fig.subplots_adjust(bottom=0.22)\n\n"
    "ecr_10s = df_10s_to_60s['ECR'].dropna()\n"
    "counts, edges, _ = ax.hist(ecr_10s, bins=50, edgecolor='white', linewidth=0.4, alpha=0.75, color=BLUE)\n\n"
    "kde = gaussian_kde(ecr_10s, bw_method=0.10)\n"
    "xs = np.linspace(ecr_10s.min(), ecr_10s.max(), 1200)\n"
    "kde_scaled = kde(xs) / kde(xs).max() * np.percentile(counts, 95)\n"
    "ax.plot(xs, kde_scaled, color=BLUE, lw=2, alpha=0.85)\n\n"
    "mean_ecr = ecr_10s.mean()\n"
    "ax.axvline(mean_ecr, color=BLUE, lw=1.5, ls='--', alpha=0.9)\n"
    "ax.text(mean_ecr + 0.01, np.percentile(counts, 95) * 0.95,\n"
    "        f'μ = {mean_ecr:.3f}', color=BLUE, fontsize=10)\n\n"
    "ax.set_xlabel('ECR — ranking normalizado de retención', fontsize=11)\n"
    "ax.set_ylabel('Número de vídeos', fontsize=11)\n"
    "ax.set_title('Distribución del ECR — Vídeos ≥ 10 s', fontsize=13, fontweight='bold', pad=10)\n"
    "ax.grid(axis='y', linestyle='--', alpha=0.35, color='grey')\n\n"
    "fig.text(0.5, 0.04,\n"
    "    f'Distribución del ECR filtrada a vídeos ≥ 10 s. '\n"
    "    f'N = {len(df_10s_to_60s):,} vídeos ({len(df_10s_to_60s)/len(df_train)*100:.1f}% del total).',\n"
    "    ha='center', va='top', fontsize=9, color='#444444', style='italic')\n\n"
    "plt.savefig('../results/ecr_dist_10sto60s.png', dpi=300, bbox_inches='tight')\n"
    "plt.show()"
))

cells_edd.append(new_markdown_cell(
    "### Análisis: distribución uniforme vs. bimodal\n\n"
    "Cuando se analiza el conjunto completo del challenge (N = 106.192 vídeos, 5–60 s),\n"
    "la distribución del ECR presenta una forma **aproximadamente uniforme** en [0, 1]\n"
    "(figura anterior). Este comportamiento contrasta con la distribución bimodal documentada\n"
    "por Li et al. sobre la primera versión del dataset SnapUGC (v1), que excluía\n"
    "explícitamente los vídeos de duración inferior a 10 s.\n\n"
    "La causa reside en una limitación semántica del ECR para vídeos muy cortos.\n"
    "El ECR mide si el usuario supera un umbral de visionado de 5 segundos, una\n"
    "**decisión consciente** de continuar viendo. Sin embargo, en vídeos de duración\n"
    "igual o próxima a ese umbral, la reproducción automática de la plataforma hace\n"
    "que el umbral se supere —o no— antes de que el usuario haya podido evaluar el\n"
    "contenido y actuar en consecuencia. El ECR de estos vídeos refleja la\n"
    "**velocidad de reacción del usuario**, no su interés real por el contenido,\n"
    "generando valores distribuidos casi aleatoriamente en [0, 1].\n\n"
    "Al filtrar a `duration ≥ 10s` (N = 49.855 vídeos), la distribución recupera\n"
    "el **patrón bimodal** característico: un primer pico en torno a ECR ≈ 0.1\n"
    "(usuarios que descartan el vídeo) y un segundo pico próximo a ECR ≈ 0.9\n"
    "(vídeos que retienen la atención). Esta polarización refleja la dinámica\n"
    "de consumo basada en *swipe* de las plataformas de vídeo corto.\n\n"
    "> **Nota metodológica:** el filtro `duration ≥ 10s` se aplica aquí únicamente\n"
    "> con fines exploratorios. Los experimentos de modelado se realizan sobre el\n"
    "> conjunto completo, respetando las condiciones oficiales del challenge."
))

cells_edd.append(new_markdown_cell("## 3. Comparativa bimodal: vídeos < 10 s vs. ≥ 10 s"))

cells_edd.append(new_code_cell(
    "ecr_short = df_train[df_train['duration'] < 10]['ECR'].dropna()\n"
    "ecr_long  = df_train[df_train['duration'] >= 10]['ECR'].dropna()\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)\n"
    "fig.subplots_adjust(bottom=0.20, wspace=0.32)\n\n"
    "for ax, data, color, title, subtitle in [\n"
    "    (axes[0], ecr_short, CORAL,\n"
    "     'Vídeos < 10 s',\n"
    "     f'N = {len(ecr_short):,} — distribución uniforme (señal ruidosa)'),\n"
    "    (axes[1], ecr_long, BLUE,\n"
    "     'Vídeos ≥ 10 s',\n"
    "     f'N = {len(ecr_long):,} — distribución bimodal (señal válida)'),\n"
    "]:\n"
    "    counts, bins, _ = ax.hist(data, bins=50, edgecolor='white', linewidth=0.4, alpha=0.75, color=color)\n"
    "    kde = gaussian_kde(data, bw_method=0.10)\n"
    "    xs = np.linspace(0, 1, 600)\n"
    "    bin_width = bins[1] - bins[0]\n"
    "    kde_scaled = kde(xs) * len(data) * bin_width\n"
    "    ax.plot(xs, kde_scaled, color=color, lw=2.2, alpha=0.9)\n"
    "    ax.axvline(data.mean(), color=color, lw=1.5, ls='--', alpha=0.9)\n"
    "    ax.text(data.mean() + 0.02, ax.get_ylim()[1] * 0.90,\n"
    "            f'μ = {data.mean():.3f}', color=color, fontsize=9.5)\n"
    "    ax.set_xlabel('ECR — ranking normalizado de retención', fontsize=10)\n"
    "    ax.set_ylabel('Número de vídeos', fontsize=10)\n"
    "    ax.set_title(title, fontsize=12, fontweight='bold')\n"
    "    ax.set_xlim(-0.02, 1.02)\n"
    "    ax.grid(axis='y', linestyle='--', alpha=0.35, color='grey')\n"
    "    ax.text(0.5, -0.15, subtitle, ha='center', transform=ax.transAxes,\n"
    "            fontsize=9, color='#555555', style='italic')\n\n"
    "fig.suptitle('Distribución del ECR por intervalo de duración',\n"
    "             fontsize=14, fontweight='bold', y=1.01)\n"
    "plt.savefig('../results/ecr_bimodal_comparison.png', dpi=300, bbox_inches='tight')\n"
    "plt.show()"
))

cells_edd.append(new_code_cell(
    "print('=' * 62)\n"
    "print('VÍDEOS CON DURACIÓN < 10 s POR PARTICIÓN')\n"
    "print('=' * 62)\n"
    "for name, df in [('train', df_train), ('val', df_val), ('test', df_test)]:\n"
    "    n_short = (df['duration'] < 10).sum()\n"
    "    pct     = n_short / len(df) * 100\n"
    "    print(f'{name:6s}: {n_short:6,} vídeos < 10s  ({pct:.1f}% del total)')\n\n"
    "print()\n"
    "print(f'ECR medio vídeos <10s  (train): {ecr_short.mean():.3f} ± {ecr_short.std():.3f}')\n"
    "print(f'ECR medio vídeos ≥10s (train): {ecr_long.mean():.3f} ± {ecr_long.std():.3f}')\n"
    "ks_s, p_s = stats.ks_2samp(ecr_short, ecr_long)\n"
    "print(f'KS test <10s vs ≥10s:  stat={ks_s:.4f},  p={p_s:.2e}')"
))

cells_edd.append(new_markdown_cell(
    "### Interpretación: ECR como señal ruidosa para vídeos < 10 s\n\n"
    "El test KS entre vídeos <10 s y ≥10 s arroja un estadístico de 0.086 con p=9.18×10⁻¹⁷² —\n"
    "evidencia estadística abrumadora de que las distribuciones de ECR son cualitativamente\n"
    "distintas. Los vídeos cortos (<10 s) presentan una distribución de ECR casi uniforme\n"
    "(media ~0.500), sin la estructura bimodal característica de los vídeos más largos\n"
    "(media ~0.495 con mayor dispersión en los extremos).\n\n"
    "La proporción de vídeos cortos (~52%) es **constante entre las tres particiones**,\n"
    "lo que confirma que es una propiedad estructural del dataset v2 y no un artefacto\n"
    "del muestreo. Esta observación es coherente con la decisión de Li et al. (ECCV 2024)\n"
    "de excluir los vídeos <10 s en la versión original del benchmark."
))

cells_edd.append(new_markdown_cell("## 4. Distribución de la duración"))

cells_edd.append(new_code_cell(
    "fig, ax = plt.subplots(figsize=(7, 5))\n"
    "fig.subplots_adjust(bottom=0.22)\n\n"
    "data_dur = df_train['duration'].dropna().values\n"
    "counts, edges, _ = ax.hist(data_dur, bins=50, edgecolor='white',\n"
    "                            linewidth=0.4, alpha=0.75, color=CORAL)\n\n"
    "# KDE en escala log (corrección jacobiana para distribución log-normal)\n"
    "log_data = np.log(data_dur)\n"
    "kde = gaussian_kde(log_data, bw_method=0.25)\n"
    "xs_log  = np.linspace(np.log(3), np.log(65), 600)\n"
    "xs_orig = np.exp(xs_log)\n"
    "kde_vals_corrected = kde(xs_log) / xs_orig\n"
    "kde_scaled = kde_vals_corrected / kde_vals_corrected.max() * counts.max() * 0.95\n"
    "ax.plot(xs_orig, kde_scaled, color=CORAL, lw=2.5, alpha=0.9)\n\n"
    "mean_dur = data_dur.mean()\n"
    "ax.axvline(mean_dur, color=CORAL, lw=1.5, ls='--', alpha=0.9)\n"
    "ax.text(mean_dur + 0.5, counts.max() * 0.92,\n"
    "        f'μ = {mean_dur:.1f}s', color=CORAL, fontsize=10)\n\n"
    "ax.set_xlabel('Duración del vídeo (segundos)', fontsize=11)\n"
    "ax.set_xlim(3, 65)\n"
    "ax.set_ylabel('Número de vídeos', fontsize=11)\n"
    "ax.set_title('Distribución de la Duración', fontsize=13, fontweight='bold', pad=10)\n"
    "ax.xaxis.set_major_locator(ticker.MultipleLocator(10))\n"
    "ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))\n"
    "ax.grid(axis='both', which='both', linestyle='--', alpha=0.25, color='grey')\n\n"
    "fig.text(0.5, 0.04,\n"
    "    f'Distribución de la duración en el dataset SnapUGC v2 '\n"
    "    f'(N = {len(df_train):,} vídeos de entrenamiento, 5–60 s).',\n"
    "    ha='center', va='top', fontsize=9, color='#444444', style='italic')\n\n"
    "plt.savefig('../results/duration_dist.png', dpi=300, bbox_inches='tight')\n"
    "plt.show()"
))

cells_edd.append(new_markdown_cell(
    "### Interpretación: Distribución de la duración\n\n"
    "La duración presenta una distribución claramente log-normal con un pico pronunciado\n"
    "en torno a los 7–10 segundos, reflejo del formato dominante de vídeo muy corto en la\n"
    "plataforma Snapchat Spotlight. La media (15.6 s) supera considerablemente a la mediana\n"
    "(9.7 s), evidenciando la cola derecha generada por los vídeos de mayor duración.\n\n"
    "Prácticamente todos los vídeos tienen duración entre 5 s y 60 s (solo 21 vídeos superan\n"
    "los 60 s, un 0.02% del total), lo que confirma que los límites declarados del dataset\n"
    "son efectivos."
))

cells_edd.append(new_markdown_cell(
    "## 5. Análisis comparativo por partición (Train / Val / Test)\n\n"
    "### 5.1 Distribución del ECR por partición (KDE overlay)"
))

cells_edd.append(new_code_cell(
    "fig, ax = plt.subplots(figsize=(9, 5))\n"
    "fig.subplots_adjust(bottom=0.20)\n\n"
    "split_ecr = [\n"
    "    (df_train['ECR'].dropna(), BLUE,  'Train (N=106,192)'),\n"
    "    (df_val['ECR'].dropna(),   CORAL, 'Val (N=6,000)'),\n"
    "    (df_test['ECR'].dropna(),  GREEN, 'Test (N=8,459)'),\n"
    "]\n\n"
    "for data, color, label in split_ecr:\n"
    "    kde = gaussian_kde(data, bw_method=0.12)\n"
    "    xs  = np.linspace(0, 1, 600)\n"
    "    ax.plot(xs, kde(xs), color=color, lw=2.2, label=label)\n"
    "    ax.axvline(data.mean(), color=color, lw=1.0, ls='--', alpha=0.7)\n\n"
    "ax.set_xlabel('ECR — ranking normalizado de retención', fontsize=12)\n"
    "ax.set_ylabel('Densidad', fontsize=12)\n"
    "ax.set_title('Distribución del ECR por partición (Train / Val / Test)',\n"
    "             fontsize=13, fontweight='bold', pad=10)\n"
    "ax.legend(fontsize=10, framealpha=0.8)\n"
    "ax.grid(axis='y', linestyle='--', alpha=0.35, color='grey')\n\n"
    "fig.text(0.5, 0.04,\n"
    "    'KDE del ECR en los tres subconjuntos oficiales del challenge ICCV VQualA 2025.\\n'\n"
    "    'Las líneas discontinuas indican la media muestral de cada partición.',\n"
    "    ha='center', va='bottom', fontsize=8.5, color='#444444', style='italic')\n\n"
    "plt.savefig('../results/ecr_kde_splits.png', dpi=300, bbox_inches='tight')\n"
    "plt.show()"
))

cells_edd.append(new_markdown_cell(
    "### Interpretación: Distribución del ECR por partición\n\n"
    "Las tres particiones presentan distribuciones de ECR muy similares: media en torno a\n"
    "0.49–0.50, desviación típica ~0.29. Los tests KS confirman que train↔val es\n"
    "estadísticamente homogéneo (p=0.172), mientras que train↔test y val↔test muestran\n"
    "diferencias significativas pero de tamaño de efecto muy pequeño (KS stat ≤ 0.027).\n"
    "Este leve sesgo del conjunto de test —media ECR 0.502 frente a 0.498 en entrenamiento—\n"
    "es una propiedad del dataset y debe tenerse en cuenta al interpretar los resultados."
))

cells_edd.append(new_markdown_cell("### 5.2 Distribución de la duración por partición (KDE overlay)"))

cells_edd.append(new_code_cell(
    "fig, ax = plt.subplots(figsize=(9, 5))\n"
    "fig.subplots_adjust(bottom=0.20)\n\n"
    "split_dur = [\n"
    "    (df_train['duration'].dropna(), BLUE,  'Train (N=106,192)'),\n"
    "    (df_val['duration'].dropna(),   CORAL, 'Val (N=6,000)'),\n"
    "    (df_test['duration'].dropna(),  GREEN, 'Test (N=8,459)'),\n"
    "]\n\n"
    "for data, color, label in split_dur:\n"
    "    log_data = np.log(data.clip(lower=1))\n"
    "    kde = gaussian_kde(log_data, bw_method=0.25)\n"
    "    xs_log  = np.linspace(np.log(4), np.log(62), 600)\n"
    "    xs_orig = np.exp(xs_log)\n"
    "    kde_vals = kde(xs_log) / xs_orig\n"
    "    kde_vals = kde_vals / kde_vals.max()\n"
    "    ax.plot(xs_orig, kde_vals, color=color, lw=2.2, label=label)\n"
    "    ax.axvline(data.mean(), color=color, lw=1.0, ls='--', alpha=0.7)\n\n"
    "ax.set_xlabel('Duración del vídeo (segundos)', fontsize=12)\n"
    "ax.set_ylabel('Densidad normalizada', fontsize=12)\n"
    "ax.set_title('Distribución de la duración por partición (Train / Val / Test)',\n"
    "             fontsize=13, fontweight='bold', pad=10)\n"
    "ax.set_xlim(4, 65)\n"
    "ax.legend(fontsize=10, framealpha=0.8)\n"
    "ax.grid(axis='y', linestyle='--', alpha=0.35, color='grey')\n\n"
    "fig.text(0.5, 0.04,\n"
    "    'KDE de la duración en los tres subconjuntos del challenge ICCV VQualA 2025.\\n'\n"
    "    'Las líneas discontinuas indican la duración media de cada partición.',\n"
    "    ha='center', va='bottom', fontsize=8.5, color='#444444', style='italic')\n\n"
    "plt.savefig('../results/duration_kde_splits.png', dpi=300, bbox_inches='tight')\n"
    "plt.show()"
))

cells_edd.append(new_markdown_cell(
    "### Interpretación: Distribución de la duración por partición\n\n"
    "La duración es completamente homogénea entre las tres particiones (todos los tests KS\n"
    "con p > 0.40). La distribución presenta una forma log-normal con un pico pronunciado\n"
    "en torno a los 7–10 segundos. El ~52% de vídeos cortos (<10 s) es constante en las\n"
    "tres particiones, confirmando que es una característica estructural del dataset v2."
))

cells_edd.append(new_markdown_cell(
    "## 6. Tests de Kolmogorov-Smirnov — Comparabilidad entre particiones\n\n"
    "**H₀**: las dos distribuciones provienen de la misma distribución de probabilidad.  \n"
    "**p < 0.05** → sesgo distribucional detectado → las particiones no son intercambiables."
))

cells_edd.append(new_code_cell(
    "pairs = [\n"
    "    ('ECR',      df_train['ECR'].dropna(),      df_val['ECR'].dropna(),       'train vs val'),\n"
    "    ('ECR',      df_train['ECR'].dropna(),      df_test['ECR'].dropna(),      'train vs test'),\n"
    "    ('ECR',      df_val['ECR'].dropna(),        df_test['ECR'].dropna(),      'val vs test'),\n"
    "    ('duration', df_train['duration'].dropna(), df_val['duration'].dropna(),  'train vs val'),\n"
    "    ('duration', df_train['duration'].dropna(), df_test['duration'].dropna(), 'train vs test'),\n"
    "    ('duration', df_val['duration'].dropna(),   df_test['duration'].dropna(), 'val vs test'),\n"
    "]\n\n"
    "print(f'{\"Variable\":10s} | {\"Comparación\":18s} | {\"KS stat\":8s} | {\"p-valor\":10s} | Veredicto')\n"
    "print('-' * 70)\n"
    "for var, a, b, pair in pairs:\n"
    "    ks_stat, p_val = stats.ks_2samp(a, b)\n"
    "    verdict = 'SESGO DETECTADO ⚠' if p_val < 0.05 else 'OK (sin sesgo) ✓'\n"
    "    print(f'{var:10s} | {pair:18s} | {ks_stat:8.4f} | {p_val:10.2e} | {verdict}')\n\n"
    "print()\n"
    "print('Nota: p < 0.05 indica distribuciones estadísticamente distintas.')"
))

cells_edd.append(new_markdown_cell(
    "### Interpretación: Tests de Kolmogorov-Smirnov\n\n"
    "Los resultados revelan un patrón asimétrico: **la partición de validación es\n"
    "estadísticamente comparable a la de entrenamiento** (ECR: p=0.172; duración: p=0.303),\n"
    "mientras que el conjunto de test muestra una desviación leve pero significativa en ECR\n"
    "(train↔test: KS=0.0165, p=0.028; val↔test: KS=0.0269, p=0.012).\n\n"
    "Sin embargo, el tamaño del efecto es mínimo: un estadístico KS de 0.027 indica que las\n"
    "distribuciones acumuladas difieren como máximo en un 2.7% en cualquier punto. La\n"
    "**duración** es homogénea en todos los pares (p > 0.30), confirmando que el sesgo en\n"
    "ECR no se explica por composiciones diferentes de tipos de vídeo."
))

cells_edd.append(new_markdown_cell(
    "## Conclusiones del análisis\n\n"
    "| Análisis | Resultado |\n"
    "|----------|-----------|\n"
    "| ECR train↔val | ✓ Homogéneo (p=0.172) |\n"
    "| ECR train↔test | ⚠ Sesgo leve (p=0.028, KS=0.017) |\n"
    "| ECR val↔test | ⚠ Sesgo leve (p=0.012, KS=0.027) |\n"
    "| Duración (todos los pares) | ✓ Homogéneo (p > 0.30) |\n"
    "| Vídeos < 10 s | ~52% en cada partición |\n"
    "| ECR < 10 s vs ≥ 10 s | ⚠ Distribuciones distintas (p ≈ 9×10⁻¹⁷²) |\n\n"
    "**Implicación para el diseño experimental:** Las métricas de evaluación se reportan\n"
    "sobre la partición de validación oficial (val), que es estadísticamente comparable\n"
    "a entrenamiento. El ligero sesgo del conjunto de test es una limitación menor del\n"
    "dataset y no invalida la comparación de modelos entre sí."
))

# ── Build notebook 1 ────────────────────────────────────────────
nb1 = new_notebook()
nb1.cells = cells_edd
nb1.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (tfg-ugc-retention)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {"name": "python", "version": "3.10.0"}
}

# ═══════════════════════════════════════════════════════════════
# NOTEBOOK 2: CorrelacionesyPredictores.ipynb
# ═══════════════════════════════════════════════════════════════

cells_cp = []

cells_cp.append(new_markdown_cell(
    "# EDA — Correlaciones y Predictores del ECR\n"
    "## Dataset SnapUGC v2 — ICCV VQualA Challenge 2025\n\n"
    "Este notebook analiza la capacidad predictiva de las variables técnicas y de metadatos\n"
    "del dataset SnapUGC v2 sobre el ECR. Se incluyen análisis de correlación (Spearman),\n"
    "visualización de la relación ECR × duración, análisis de metadatos de texto, y\n"
    "el baseline Ridge B0 como cota inferior de referencia.\n\n"
    "**Objetivo:** demostrar cuantitativamente que las características técnicas y de\n"
    "metadatos tienen un poder predictivo muy limitado, justificando la necesidad del\n"
    "enfoque multimodal de la pipeline del TFG."
))

cells_cp.append(new_code_cell(
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.ticker as ticker\n"
    "import seaborn as sns\n"
    "from scipy.stats import spearmanr, pearsonr, gaussian_kde\n"
    "from sklearn.linear_model import Ridge\n"
    "from sklearn.metrics import mean_squared_error\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n\n"
    "plt.rcParams.update({\n"
    "    'font.family': 'serif',\n"
    "    'axes.spines.top': False,\n"
    "    'axes.spines.right': False,\n"
    "    'axes.facecolor': '#F9F9F9',\n"
    "    'figure.facecolor': 'white',\n"
    "})\n\n"
    "BASE_RAW  = '../data/raw/'\n"
    "BASE_PROC = '../data/processed/'\n"
    "BLUE   = '#2E6FA3'\n"
    "CORAL  = '#D95F4B'\n"
    "GREEN  = '#4A9B7F'"
))

cells_cp.append(new_code_cell(
    "# ── TRAIN (exploración + entrenamiento del baseline)\n"
    "train_data = pd.read_csv(BASE_RAW + 'train_data.csv')\n"
    "train_meta = pd.read_csv(BASE_PROC + 'train_metadata.csv')\n"
    "df = train_data.merge(train_meta, on='Id', how='inner')\n"
    "df['has_title']       = df['Title'].notna() & (df['Title'].str.strip() != '')\n"
    "df['has_description'] = df['Description'].notna() & (df['Description'].str.strip() != '')\n"
    "df['title_length']    = df['Title'].fillna('').str.len()\n"
    "df['aspect_ratio']    = df['width'] / df['height']\n"
    "df['is_vertical']     = df['height'] > df['width']\n"
    "df['resolution']      = df['width'] * df['height']\n\n"
    "# ── VAL oficial (evaluación del baseline B0)\n"
    "val_data = pd.read_csv(BASE_RAW + 'val_data.csv')\n"
    "val_meta = pd.read_csv(BASE_PROC + 'val_metadata.csv')\n"
    "df_val = val_data.merge(val_meta, on='Id', how='inner')\n"
    "df_val['has_title']       = df_val['Title'].notna() & (df_val['Title'].str.strip() != '')\n"
    "df_val['has_description'] = df_val['Description'].notna() & (df_val['Description'].str.strip() != '')\n"
    "df_val['title_length']    = df_val['Title'].fillna('').str.len()\n"
    "df_val['resolution']      = df_val['width'] * df_val['height']\n\n"
    "print(f'Train: {len(df):,} | Val: {len(df_val):,}')\n"
    "print(f'\\nis_vertical (train): {df[\"is_vertical\"].mean()*100:.2f}% — near-constant (excluida)')\n"
    "print(f'has_title (train):   {df[\"has_title\"].mean()*100:.1f}%')\n"
    "print(f'has_desc (train):    {df[\"has_description\"].mean()*100:.1f}%')"
))

cells_cp.append(new_markdown_cell(
    "## 1. Análisis de correlaciones de Spearman\n\n"
    "La correlación de Spearman es la métrica apropiada aquí porque el ECR es una variable\n"
    "de ranking normalizada (no una probabilidad directa), por lo que las relaciones\n"
    "monótonas no-lineales son más relevantes que las correlaciones lineales de Pearson.\n\n"
    "> **Nota:** `is_vertical` se excluye del análisis porque el 99.93% de los vídeos son\n"
    "> verticales — varianza prácticamente nula, lo que produciría correlaciones espurias."
))

cells_cp.append(new_code_cell(
    "cols_to_correlate = ['ECR', 'duration', 'has_title', 'has_description', 'title_length', 'resolution']\n"
    "corr_matrix = df[cols_to_correlate].corr(method='spearman')\n\n"
    "plt.figure(figsize=(10, 8))\n"
    "sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,\n"
    "            square=True, linewidths=.5, cbar_kws={'shrink': .75})\n"
    "plt.title('Variables técnicas vs ECR: Correlación de Spearman', fontsize=14, pad=20)\n"
    "plt.tight_layout()\n"
    "plt.savefig('../results/spearman_correlation_heatmap.png', dpi=300, bbox_inches='tight')\n"
    "plt.show()\n\n"
    "print('\\nCorrelaciones con ECR (Spearman):')\n"
    "ecr_corr = corr_matrix['ECR'].drop('ECR').sort_values(key=abs, ascending=False)\n"
    "for feat, corr in ecr_corr.items():\n"
    "    print(f'  {feat:20s}: {corr:+.4f}')"
))

cells_cp.append(new_markdown_cell(
    "### Interpretación: Correlaciones de Spearman\n\n"
    "Todas las correlaciones con ECR son muy débiles (|ρ| < 0.10). La variable con mayor\n"
    "correlación absoluta es `duration` (ρ ≈ -0.04), seguida de `resolution` y las\n"
    "variables de metadatos textuales. Este resultado confirma que las características\n"
    "técnicas por sí solas **no tienen capacidad predictiva significativa** sobre el ECR.\n\n"
    "La correlación negativa de `duration` con ECR es coherente con la hipótesis de ruido\n"
    "de señal: los vídeos más cortos (<10 s) tienen ECR distribuido casi uniformemente,\n"
    "lo que no aporta señal, mientras que los vídeos más largos muestran el patrón bimodal\n"
    "verdadero con tendencia hacia valores más extremos."
))

cells_cp.append(new_markdown_cell(
    "## 2. Relación ECR × Duración — Análisis de densidad\n\n"
    "Un *hexbin plot* permite visualizar la densidad conjunta en el espacio duración × ECR,\n"
    "revelando la verdadera distribución volumétrica de la retención frente al tiempo de vídeo."
))

cells_cp.append(new_code_cell(
    "df_clean = df[(df['duration'] >= 5) & (df['duration'] <= 60)].copy()\n\n"
    "fig, ax = plt.subplots(figsize=(10, 6))\n"
    "fig.subplots_adjust(bottom=0.18, right=0.88)\n\n"
    "hb = ax.hexbin(df_clean['duration'], df_clean['ECR'],\n"
    "               gridsize=45, cmap='YlOrRd', bins='log', mincnt=1,\n"
    "               linewidths=0.15, edgecolors='white', alpha=0.92)\n\n"
    "cbar_ax = fig.add_axes([0.90, 0.18, 0.022, 0.72])\n"
    "cb = fig.colorbar(hb, cax=cbar_ax)\n"
    "cb.set_label('Número de vídeos (escala log)', fontsize=10, color='#444444')\n"
    "cb.ax.tick_params(labelsize=9, colors='#444444')\n"
    "cb.outline.set_visible(False)\n\n"
    "# Línea de tendencia (media de ECR por ventana de duración)\n"
    "bins_dur = np.arange(5, 61, 2)\n"
    "bin_centers, bin_means, bin_stds = [], [], []\n"
    "for i in range(len(bins_dur) - 1):\n"
    "    mask = (df_clean['duration'] >= bins_dur[i]) & (df_clean['duration'] < bins_dur[i+1])\n"
    "    subset = df_clean.loc[mask, 'ECR']\n"
    "    if len(subset) > 10:\n"
    "        bin_centers.append((bins_dur[i] + bins_dur[i+1]) / 2)\n"
    "        bin_means.append(subset.mean())\n"
    "        bin_stds.append(subset.std())\n\n"
    "bin_centers = np.array(bin_centers)\n"
    "bin_means   = np.array(bin_means)\n"
    "bin_stds    = np.array(bin_stds)\n\n"
    "ax.plot(bin_centers, bin_means, color='white', lw=3.0, zorder=5, alpha=0.9)\n"
    "ax.plot(bin_centers, bin_means, color=BLUE, lw=1.8, zorder=6, alpha=0.95,\n"
    "        label='ECR medio por ventana')\n"
    "ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds,\n"
    "                color=BLUE, alpha=0.12, zorder=4, label='±1 desv. típica')\n\n"
    "ax.axhline(0.5, color='#444444', lw=1.0, ls=':', alpha=0.6, zorder=3)\n"
    "ax.text(61.5, 0.505, 'ECR = 0.5', fontsize=8.5, color='#444444', va='bottom')\n"
    "mean_ecr_global = df_clean['ECR'].mean()\n"
    "ax.axhline(mean_ecr_global, color=CORAL, lw=1.2, ls='--', alpha=0.8, zorder=3)\n"
    "ax.text(61.5, mean_ecr_global + 0.01,\n"
    "        f'μ global = {mean_ecr_global:.3f}', fontsize=8.5, color=CORAL, va='bottom')\n\n"
    "ax.set_xlabel('Duración del vídeo (segundos)', fontsize=12)\n"
    "ax.set_ylabel('ECR — ranking normalizado de retención', fontsize=12)\n"
    "ax.set_title('Relación entre Duración y ECR\\n'\n"
    "             'Densidad de vídeos en el espacio duración × retención',\n"
    "             fontsize=13, fontweight='bold', pad=12)\n"
    "ax.set_xlim(4, 62)\n"
    "ax.set_ylim(-0.02, 1.05)\n"
    "ax.xaxis.set_major_locator(ticker.MultipleLocator(10))\n"
    "ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))\n"
    "ax.grid(axis='both', linestyle='--', alpha=0.25, color='grey', zorder=0)\n"
    "ax.legend(loc='upper right', fontsize=9, framealpha=0.7, edgecolor='#cccccc', fancybox=False)\n\n"
    "fig.text(0.5, 0.02,\n"
    "    f'Análisis de densidad ECR × Duración. N = {len(df_clean):,} vídeos filtrados entre 5–60 s.',\n"
    "    ha='center', va='bottom', fontsize=8.5, color='#444444', style='italic')\n\n"
    "plt.savefig('../results/hexbin_ecr_duration.png', dpi=300, bbox_inches='tight')\n"
    "plt.show()"
))

cells_cp.append(new_markdown_cell(
    "### Interpretación: Relación ECR × Duración\n\n"
    "El hexbin plot confirma visualmente la hipótesis del ruido de señal para vídeos cortos:\n\n"
    "- **5–10 s:** alta densidad con ECR distribuido uniformemente (banda horizontal difusa).\n"
    "  La línea de tendencia oscila cerca de ECR = 0.5 con alta desviación típica,\n"
    "  evidenciando la ausencia de señal coherente.\n"
    "- **10–30 s:** la densidad se concentra progresivamente en los extremos (ECR < 0.2 y\n"
    "  ECR > 0.8), apareciendo el patrón bimodal. La tendencia converge hacia ECR ≈ 0.50\n"
    "  con menor dispersión.\n"
    "- **30–60 s:** la polarización bimodal es más pronunciada. Los vídeos muy largos tienden\n"
    "  a tener ECR más bajo, posiblemente por agotamiento de la atención del espectador.\n\n"
    "La correlación Spearman `duration` × ECR es estadísticamente significativa pero de\n"
    "tamaño de efecto muy pequeño (ρ ≈ -0.04), consistente con lo observado en el heatmap."
))

cells_cp.append(new_markdown_cell("## 3. Composición del dataset por formato de vídeo"))

cells_cp.append(new_code_cell(
    "n_h   = (df['is_vertical'] == 0).sum()\n"
    "n_v   = (df['is_vertical'] == 1).sum()\n"
    "total = len(df)\n"
    "pct_v = n_v / total * 100\n"
    "pct_h = n_h / total * 100\n\n"
    "print(f'Vídeos verticales:   {n_v:,} ({pct_v:.2f}%)')\n"
    "print(f'Vídeos horizontales: {n_h:,} ({pct_h:.2f}%)')\n\n"
    "plot_pct_h = max(pct_h, 0.45)\n"
    "plot_pct_v = 100 - plot_pct_h\n\n"
    "fig, ax = plt.subplots(figsize=(7, 6))\n"
    "fig.subplots_adjust(bottom=0.12)\n"
    "ax.set_aspect('equal')\n"
    "ax.axis('off')\n\n"
    "wedges, _ = ax.pie(\n"
    "    [plot_pct_h, plot_pct_v],\n"
    "    colors=[GREEN, CORAL],\n"
    "    startangle=90,\n"
    "    counterclock=False,\n"
    "    explode=[0.08, 0.0],\n"
    "    wedgeprops=dict(width=0.45, edgecolor='white', linewidth=1.5),\n"
    "    radius=1.0\n"
    ")\n\n"
    "ax.text(0,  0.10, f'{pct_v:.2f}%', ha='center', va='center',\n"
    "        fontsize=30, fontweight='bold', color=CORAL)\n"
    "ax.text(0, -0.18, 'Contenido vertical', ha='center', va='center',\n"
    "        fontsize=10, color=CORAL, style='italic')\n"
    "ax.annotate(f'Vertical — {n_v:,} vídeos',\n"
    "    xy=(0.75, 0.75), xytext=(1.40, 1.20), ha='center', fontsize=9.5,\n"
    "    color=CORAL, fontweight='bold',\n"
    "    arrowprops=dict(arrowstyle='->', color=CORAL, lw=1.0,\n"
    "                    connectionstyle='arc3,rad=-0.2'))\n"
    "ax.annotate(f'Horizontal — {n_h:,} vídeos  ({pct_h:.2f}%)',\n"
    "    xy=(0.04, 1.05), xytext=(-1.40, 1.20), ha='center', fontsize=9.5,\n"
    "    color=GREEN, fontweight='bold',\n"
    "    arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.0,\n"
    "                    connectionstyle='arc3,rad=-0.2'))\n\n"
    "ax.set_title('Composición del Dataset por Formato de Vídeo',\n"
    "             fontsize=13, fontweight='bold', pad=16, y=1.06)\n"
    "fig.text(0.5, 0.04,\n"
    "    f'Dataset SnapUGC v2 (N = {total:,} vídeos de entrenamiento).',\n"
    "    ha='center', va='bottom', fontsize=8.5, color='#555555', style='italic')\n\n"
    "plt.savefig('../results/horizontal_vs_vertical_donut.png', dpi=300, bbox_inches='tight')\n"
    "plt.show()"
))

cells_cp.append(new_markdown_cell(
    "### Interpretación: Composición por formato\n\n"
    "El 99.93% de los vídeos de SnapUGC son verticales — una distribución prácticamente\n"
    "constante. Esta cuasi-constancia hace que `is_vertical` sea **inútil como predictor**\n"
    "(varianza casi nula → correlación Spearman ≈ 0 espuria) y por ello se excluye del\n"
    "análisis de correlaciones y del baseline Ridge.\n\n"
    "Este resultado es consistente con la naturaleza de la plataforma Snapchat, diseñada\n"
    "exclusivamente para consumo móvil en orientación vertical."
))

cells_cp.append(new_markdown_cell("## 4. Metadatos de texto: influencia de título y descripción"))

cells_cp.append(new_code_cell(
    "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n"
    "fig.subplots_adjust(bottom=0.20, wspace=0.35)\n\n"
    "# ── Panel 1: ECR por has_title\n"
    "ax = axes[0]\n"
    "ecr_no_title  = df[df['has_title'] == False]['ECR']\n"
    "ecr_yes_title = df[df['has_title'] == True]['ECR']\n"
    "ax.boxplot([ecr_no_title, ecr_yes_title], labels=['Sin título', 'Con título'],\n"
    "           patch_artist=True, boxprops=dict(facecolor=BLUE, alpha=0.6),\n"
    "           medianprops=dict(color='white', lw=2))\n"
    "srcc_title, p_title = spearmanr(df['has_title'].astype(int), df['ECR'])\n"
    "ax.set_title(f'ECR por presencia de título\\nρ = {srcc_title:.4f}, p = {p_title:.2e}',\n"
    "             fontsize=10, fontweight='bold')\n"
    "ax.set_ylabel('ECR', fontsize=10)\n"
    "ax.grid(axis='y', linestyle='--', alpha=0.35)\n"
    "ax.text(0.5, 0.02, f'Sin título: N={len(ecr_no_title):,}\\nCon título: N={len(ecr_yes_title):,}',\n"
    "        ha='center', transform=ax.transAxes, fontsize=8, color='#555555')\n\n"
    "# ── Panel 2: ECR por has_description\n"
    "ax = axes[1]\n"
    "ecr_no_desc  = df[df['has_description'] == False]['ECR']\n"
    "ecr_yes_desc = df[df['has_description'] == True]['ECR']\n"
    "ax.boxplot([ecr_no_desc, ecr_yes_desc], labels=['Sin desc.', 'Con desc.'],\n"
    "           patch_artist=True, boxprops=dict(facecolor=CORAL, alpha=0.6),\n"
    "           medianprops=dict(color='white', lw=2))\n"
    "srcc_desc, p_desc = spearmanr(df['has_description'].astype(int), df['ECR'])\n"
    "ax.set_title(f'ECR por presencia de descripción\\nρ = {srcc_desc:.4f}, p = {p_desc:.2e}',\n"
    "             fontsize=10, fontweight='bold')\n"
    "ax.set_ylabel('ECR', fontsize=10)\n"
    "ax.grid(axis='y', linestyle='--', alpha=0.35)\n"
    "ax.text(0.5, 0.02, f'Sin desc.: N={len(ecr_no_desc):,}\\nCon desc.: N={len(ecr_yes_desc):,}',\n"
    "        ha='center', transform=ax.transAxes, fontsize=8, color='#555555')\n\n"
    "# ── Panel 3: ECR vs longitud del título\n"
    "ax = axes[2]\n"
    "df_with_title = df[df['has_title']].copy()\n"
    "ax.scatter(df_with_title['title_length'], df_with_title['ECR'],\n"
    "           alpha=0.02, s=1.5, color=GREEN, rasterized=True)\n"
    "bins = np.arange(0, 201, 10)\n"
    "bin_c, bin_m = [], []\n"
    "for i in range(len(bins)-1):\n"
    "    mask = (df_with_title['title_length'] >= bins[i]) & (df_with_title['title_length'] < bins[i+1])\n"
    "    if mask.sum() > 20:\n"
    "        bin_c.append((bins[i]+bins[i+1])/2)\n"
    "        bin_m.append(df_with_title.loc[mask, 'ECR'].mean())\n"
    "ax.plot(bin_c, bin_m, color=GREEN, lw=2.5, label='ECR medio por ventana')\n"
    "srcc_len, p_len = spearmanr(df_with_title['title_length'], df_with_title['ECR'])\n"
    "ax.set_title(f'ECR vs longitud del título\\nρ = {srcc_len:.4f}, p = {p_len:.2e}',\n"
    "             fontsize=10, fontweight='bold')\n"
    "ax.set_xlabel('Longitud del título (caracteres)', fontsize=9)\n"
    "ax.set_ylabel('ECR', fontsize=10)\n"
    "ax.set_xlim(0, 200)\n"
    "ax.legend(fontsize=8, framealpha=0.8)\n"
    "ax.grid(axis='both', linestyle='--', alpha=0.25)\n\n"
    "fig.suptitle('Influencia de los metadatos de texto sobre el ECR',\n"
    "             fontsize=13, fontweight='bold', y=1.02)\n"
    "plt.savefig('../results/ecr_text_metadata.png', dpi=300, bbox_inches='tight')\n"
    "plt.show()"
))

cells_cp.append(new_markdown_cell(
    "### Interpretación: Metadatos de texto\n\n"
    "Las correlaciones de los metadatos de texto con el ECR son todas prácticamente nulas\n"
    "(|ρ| < 0.05). La presencia de título o descripción no tiene efecto sistemático sobre\n"
    "la retención del espectador. La longitud del título tampoco muestra una relación\n"
    "monótona apreciable.\n\n"
    "Esto sugiere que el texto en sí mismo (como variable binaria o de longitud) no es\n"
    "informativo sobre el engagement. Sin embargo, el **contenido semántico** del texto\n"
    "sí puede ser relevante — ese es el rol del encoder CLIP text en la pipeline multimodal."
))

cells_cp.append(new_markdown_cell(
    "## 5. Baseline Ridge — Predictor técnico (B0)\n\n"
    "El objetivo de este baseline es establecer una **cota inferior cuantitativa** del\n"
    "rendimiento: si un modelo Ridge entrenado con las 5 características técnicas disponibles\n"
    "consigue un Score de ~0.186 en validación oficial, cualquier modelo serio del TFG\n"
    "debe superar esta cota con amplitud.\n\n"
    "**Features:** `duration`, `has_title`, `has_description`, `title_length`, `resolution`.  \n"
    "**Excluida:** `is_vertical` (99.93% vertical → varianza nula).  \n"
    "**Evaluación:** entrenamiento en train completo → evaluación en val oficial (N=6,000)."
))

cells_cp.append(new_code_cell(
    "features = ['duration', 'has_title', 'has_description', 'title_length', 'resolution']\n\n"
    "X_train = df[features].fillna(0).astype(float)\n"
    "y_train = df['ECR'].fillna(0).astype(float)\n"
    "X_val   = df_val[features].fillna(0).astype(float)\n"
    "y_val   = df_val['ECR'].fillna(0).astype(float)\n\n"
    "model = Ridge(alpha=1.0)\n"
    "model.fit(X_train, y_train)\n"
    "y_pred = model.predict(X_val)\n\n"
    "srcc_val, _ = spearmanr(y_pred, y_val)\n"
    "plcc_val, _ = pearsonr(y_pred, y_val)\n"
    "rmse_val    = np.sqrt(mean_squared_error(y_val, y_pred))\n"
    "score       = 0.6 * srcc_val + 0.4 * plcc_val\n\n"
    "print('=== BASELINE RIDGE B0 — RESULTADOS OFICIALES (train→val) ===')\n"
    "print(f'SRCC (Spearman):  {srcc_val:.4f}')\n"
    "print(f'PLCC (Pearson):   {plcc_val:.4f}')\n"
    "print(f'RMSE:             {rmse_val:.4f}')\n"
    "print(f'Score challenge:  {score:.4f}  (= 0.6×SRCC + 0.4×PLCC)')\n"
    "print('=' * 50)\n\n"
    "# Feature importance plot\n"
    "coef_df = pd.DataFrame({'Feature': features, 'Coeficiente': model.coef_})\\\n"
    "    .sort_values(by='Coeficiente', key=abs, ascending=True)\n\n"
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "colors = [CORAL if c < 0 else BLUE for c in coef_df['Coeficiente']]\n"
    "ax.barh(coef_df['Feature'], coef_df['Coeficiente'],\n"
    "        color=colors, edgecolor='white', height=0.6)\n"
    "ax.axvline(0, color='#444444', lw=1.0, ls='--', alpha=0.7)\n"
    "ax.set_xlabel('Coeficiente Ridge', fontsize=11)\n"
    "ax.set_title('Importancia de características — Baseline Ridge B0',\n"
    "             fontsize=12, fontweight='bold')\n"
    "ax.grid(axis='x', linestyle='--', alpha=0.35)\n"
    "for i, (feat, coef) in enumerate(zip(coef_df['Feature'], coef_df['Coeficiente'])):\n"
    "    offset = 0.00002 if coef >= 0 else -0.00002\n"
    "    ax.text(coef + offset, i, f'{coef:.5f}', va='center',\n"
    "            ha='left' if coef >= 0 else 'right', fontsize=9)\n"
    "fig.tight_layout()\n"
    "plt.savefig('../results/b0_feature_importance.png', dpi=300, bbox_inches='tight')\n"
    "plt.show()"
))

cells_cp.append(new_markdown_cell(
    "### Interpretación: Baseline Ridge B0\n\n"
    "| Métrica | B0 (Ridge técnico) | C1 (pipeline EVQA) | Ganador VQualA 2025 |\n"
    "|---------|--------------------|--------------------|---------------------|\n"
    "| SRCC    | 0.1814             | ~0.62              | ~0.70               |\n"
    "| PLCC    | 0.1930             | ~0.63              | ~0.71               |\n"
    "| Score   | 0.186              | ~0.660             | ~0.703              |\n\n"
    "El baseline Ridge consigue un Score de 0.186, que sirve como **cota inferior** del TFG.\n"
    "La gap con la pipeline clásica EVQA (Score ≈ 0.660) es de +0.474 puntos — confirmando\n"
    "que el contenido visual y audiovisual es la fuente de señal dominante para predecir el ECR.\n\n"
    "`duration` es la variable más importante (coeficiente negativo: vídeos más largos →\n"
    "mayor dispersión de ECR → predicciones más cercanas a la media). Las variables de texto\n"
    "(`has_title`, `has_description`, `title_length`) tienen coeficientes prácticamente nulos,\n"
    "consistente con las correlaciones Spearman observadas."
))

cells_cp.append(new_markdown_cell(
    "## Conclusiones: poder predictivo de las variables técnicas\n\n"
    "| Variable | Correlación Spearman con ECR | Conclusión |\n"
    "|----------|------------------------------|------------|\n"
    "| `duration` | ρ ≈ −0.04 | Leve señal negativa (ruido <10 s) |\n"
    "| `resolution` | ρ ≈ +0.02 | Sin señal significativa |\n"
    "| `has_title` | ρ ≈ +0.02 | Sin señal significativa |\n"
    "| `has_description` | ρ ≈ +0.02 | Sin señal significativa |\n"
    "| `title_length` | ρ ≈ +0.01 | Sin señal significativa |\n"
    "| `is_vertical` | excluida | Cuasi-constante (99.93% vertical) |\n\n"
    "**Conclusión principal:** Las variables técnicas y de metadatos tienen un poder predictivo\n"
    "despreciable sobre el ECR. El Baseline Ridge (B0) obtiene un Score de 0.186 en validación\n"
    "oficial, estableciendo la cota inferior del TFG. Para superar esta cota es imprescindible\n"
    "analizar el contenido audiovisual y semántico de los vídeos — objetivo de los experimentos\n"
    "C1–C3 (pipeline clásica EVQA) y L1–E2 (LMMs y ensemble)."
))

# ── Build notebook 2 ────────────────────────────────────────────
nb2 = new_notebook()
nb2.cells = cells_cp
nb2.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (tfg-ugc-retention)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {"name": "python", "version": "3.10.0"}
}

# ── Write both notebooks ─────────────────────────────────────────
out1 = os.path.join(NB_DIR, 'EstadísticasDescriptivasyDistribuciones.ipynb')
out2 = os.path.join(NB_DIR, 'CorrelacionesyPredictores.ipynb')

with open(out1, 'w', encoding='utf-8') as f:
    nbformat.write(nb1, f)
print(f'✓ Written: {out1}')

with open(out2, 'w', encoding='utf-8') as f:
    nbformat.write(nb2, f)
print(f'✓ Written: {out2}')

print(f'\nNotebook 1 cells: {len(nb1.cells)}')
print(f'Notebook 2 cells: {len(nb2.cells)}')
