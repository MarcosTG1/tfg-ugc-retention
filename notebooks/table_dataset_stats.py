"""
Genera la tabla de estadísticas descriptivas del dataset (Train / Val / Test)
para el TFG. Guarda la figura como 'fig_dataset_stats.png'.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import os

# ── Estética global ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': '#F9F9F9',
    'figure.facecolor': 'white',
})

BASE = os.path.join(os.path.dirname(__file__), '..', 'data')

# ── Carga de datos ───────────────────────────────────────────────────────────
# Processed metadata (duration, width, height…)
train_proc = pd.read_csv(os.path.join(BASE, 'processed', 'train_metadata.csv'))
val_proc   = pd.read_csv(os.path.join(BASE, 'processed', 'val_metadata.csv'))
test_proc  = pd.read_csv(os.path.join(BASE, 'processed', 'test_metadata.csv'))

# Raw data (Title, Description) + ECR ground-truth
train_raw  = pd.read_csv(os.path.join(BASE, 'raw', 'train_data.csv'))
val_raw    = pd.read_csv(os.path.join(BASE, 'raw', 'val_data.csv'))
test_raw   = pd.read_csv(os.path.join(BASE, 'raw', 'test_data.csv'))

val_truth  = pd.read_csv(os.path.join(BASE, 'raw', 'val_truth.csv'))
test_truth = pd.read_csv(os.path.join(BASE, 'raw', 'test_truth.csv'))

# Merge ECR into val / test
val_raw  = val_raw.merge(val_truth[['Id', 'ECR']],  on='Id', how='left')
test_raw = test_raw.merge(test_truth[['Id', 'ECR']], on='Id', how='left')

# Merge duration into raw dataframes
train_df = train_raw.merge(train_proc[['Id', 'duration']], on='Id', how='left')
val_df   = val_raw.merge(val_proc[['Id', 'duration']],    on='Id', how='left')
test_df  = test_raw.merge(test_proc[['Id', 'duration']],  on='Id', how='left')

# ── Función auxiliar para estadísticos ──────────────────────────────────────
def ecr_stats(s: pd.Series):
    return dict(mean=s.mean(), median=s.median(), std=s.std(),
                min=s.min(), max=s.max())

def dur_stats(s: pd.Series):
    return dict(mean=s.mean(), median=s.median(), std=s.std(),
                min=s.min(), max=s.max())

def pct_nonempty(s: pd.Series) -> float:
    return (s.notna() & (s.str.strip() != '')).mean() * 100

# ── Calcular estadísticos ────────────────────────────────────────────────────
sets = {
    'Train': train_df,
    'Val':   val_df,
    'Test':  test_df,
}

stats = {}
for name, df in sets.items():
    ecr = ecr_stats(df['ECR'].dropna())
    dur = dur_stats(df['duration'].dropna())
    stats[name] = {
        'N vídeos': f"{len(df):,}",
        # ECR
        'ECR media':   f"{ecr['mean']:.3f}",
        'ECR mediana': f"{ecr['median']:.3f}",
        'ECR std':     f"{ecr['std']:.3f}",
        'ECR min':     f"{ecr['min']:.3f}",
        'ECR max':     f"{ecr['max']:.3f}",
        # Duración
        'Dur media (s)':   f"{dur['mean']:.1f}",
        'Dur mediana (s)': f"{dur['median']:.1f}",
        'Dur std (s)':     f"{dur['std']:.1f}",
        'Dur min (s)':     f"{dur['min']:.1f}",
        'Dur max (s)':     f"{dur['max']:.1f}",
        # Metadatos textuales
        '% título no vacío':      f"{pct_nonempty(df['Title']):.1f}%",
        '% descripción no vacía': f"{pct_nonempty(df['Description']):.1f}%",
    }

# ── Construir tabla ──────────────────────────────────────────────────────────
# Filas agrupadas para presentación TFG
row_groups = [
    # (etiqueta_fila, claves_en_stats)
    ("N vídeos",                               ["N vídeos"]),
    ("ECR — media / mediana / std / min / max",
     ["ECR media", "ECR mediana", "ECR std", "ECR min", "ECR max"]),
    ("Duración (s) — media / mediana / std / min / max",
     ["Dur media (s)", "Dur mediana (s)", "Dur std (s)", "Dur min (s)", "Dur max (s)"]),
    ("% vídeos con título no vacío",           ["% título no vacío"]),
    ("% vídeos con descripción no vacía",      ["% descripción no vacía"]),
]

# Crear la tabla plana con separadores de grupo
col_names = ['Estadístico', 'Train', 'Val', 'Test']
rows = []

for group_label, keys in row_groups:
    if len(keys) == 1:
        key = keys[0]
        rows.append([group_label,
                     stats['Train'][key],
                     stats['Val'][key],
                     stats['Test'][key]])
    else:
        # Fila de cabecera del grupo
        rows.append([group_label, '', '', ''])
        # Sub-filas con indentación visual
        sub_labels = {
            "ECR media":   "  media",
            "ECR mediana": "  mediana",
            "ECR std":     "  desv. típica",
            "ECR min":     "  mínimo",
            "ECR max":     "  máximo",
            "Dur media (s)":   "  media",
            "Dur mediana (s)": "  mediana",
            "Dur std (s)":     "  desv. típica",
            "Dur min (s)":     "  mínimo",
            "Dur max (s)":     "  máximo",
        }
        for key in keys:
            rows.append([sub_labels.get(key, key),
                         stats['Train'][key],
                         stats['Val'][key],
                         stats['Test'][key]])

# ── Paleta de colores ────────────────────────────────────────────────────────
C_HEADER_BG  = '#2C3E50'   # azul oscuro cabecera
C_HEADER_FG  = 'white'
C_GROUP_BG   = '#D5DCE4'   # gris azulado para filas de grupo
C_ODD_BG     = '#F9F9F9'
C_EVEN_BG    = '#EDF0F4'
C_BORDER     = '#B0B8C5'
C_TEXT       = '#1C2833'
C_ACCENT     = '#2980B9'   # azul para los valores

# Identificar qué filas son "cabecera de grupo" (valores vacíos)
is_group_row = [
    r[1] == '' and r[2] == '' and r[3] == ''
    for r in rows
]

# ── Figura ───────────────────────────────────────────────────────────────────
n_rows = len(rows)
row_h  = 0.42          # altura de cada fila (pulgadas)
fig_h  = row_h * (n_rows + 2.5)  # + espacio título y márgenes
fig_w  = 11

fig, ax = plt.subplots(figsize=(fig_w, fig_h),
                        facecolor='white')
ax.set_axis_off()

# Posiciones de columnas (normalizado 0–1)
col_x = [0.0, 0.55, 0.70, 0.85]
col_w = [0.55, 0.15, 0.15, 0.15]

def draw_cell(ax, x, y, w, h, text, bg, fg, fontsize=10,
              bold=False, ha='left', indent=0):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="square,pad=0",
        linewidth=0.4, edgecolor=C_BORDER,
        facecolor=bg, transform=ax.transData,
        clip_on=False, zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        x + indent + (0.01 if ha == 'left' else w / 2),
        y + h / 2, text,
        va='center', ha=ha,
        fontsize=fontsize,
        color=fg,
        fontweight='bold' if bold else 'normal',
        transform=ax.transData,
        clip_on=False, zorder=3,
    )

# Coordenadas: y=0 es la parte superior de cada fila
ax.set_xlim(0, 1)
ax.set_ylim(-(n_rows + 1) * row_h, row_h)

# ── Cabecera ─────────────────────────────────────────────────────────────────
hdr_labels = col_names
hdr_bold   = [True] * 4
for ci, (x, w, lbl) in enumerate(zip(col_x, col_w, hdr_labels)):
    ha = 'left' if ci == 0 else 'center'
    draw_cell(ax, x, 0, w, row_h, lbl, C_HEADER_BG, C_HEADER_FG,
              fontsize=11, bold=True, ha=ha, indent=0.01 if ci == 0 else 0)

# ── Filas de datos ───────────────────────────────────────────────────────────
for ri, row in enumerate(rows):
    y = -(ri + 1) * row_h
    grp = is_group_row[ri]

    if grp:
        bg = C_GROUP_BG
        fg_main = C_TEXT
        bold_main = True
        fs = 10
    else:
        bg = C_ODD_BG if ri % 2 == 0 else C_EVEN_BG
        fg_main = C_TEXT
        bold_main = False
        fs = 9.5

    for ci, (x, w, val) in enumerate(zip(col_x, col_w, row)):
        ha = 'left' if ci == 0 else 'center'
        indent = 0.02 if (ci == 0 and row[0].startswith('  ')) else 0.01
        fg = C_ACCENT if (ci > 0 and not grp and val != '') else fg_main
        draw_cell(ax, x, y, w, row_h, val, bg, fg,
                  fontsize=fs, bold=(bold_main and ci == 0),
                  ha=ha, indent=indent)

# ── Título ───────────────────────────────────────────────────────────────────
fig.text(0.5, 0.97,
         "Estadísticas descriptivas del dataset",
         ha='center', va='top',
         fontsize=15, fontweight='bold', color=C_HEADER_BG,
         fontfamily='serif')
fig.text(0.5, 0.93,
         r"Particiones Train / Val / Test  ·  $N_\mathrm{total}=" +
         f"{len(train_df)+len(val_df)+len(test_df):,}" + r"$ vídeos",
         ha='center', va='top',
         fontsize=10, color='#5D6D7E',
         fontfamily='serif')

plt.tight_layout(rect=[0, 0, 1, 0.92])

# Guardar
out_path = os.path.join(os.path.dirname(__file__), 'fig_dataset_stats.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight',
            facecolor='white')
print(f"Figura guardada en: {out_path}")
plt.show()
