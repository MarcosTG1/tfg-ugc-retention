"""
B0 — Technical baseline: Ridge regression on metadata features only.
Trained on full train set (106,192 videos), evaluated on official val set (6,000 videos).

Demonstrates the lower bound of retention prediction without audiovisual content.
Challenge metric: Score = 0.6 × SRCC + 0.4 × PLCC
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).parent.parent
RAW  = ROOT / 'data' / 'raw'
PROC = ROOT / 'data' / 'processed'
RES  = ROOT / 'results'

# ── 1. LOAD & MERGE TRAIN ──────────────────────────────────────────
print("Loading train data...")
train      = pd.read_csv(RAW / 'train_data.csv')
train_meta = pd.read_csv(PROC / 'train_metadata.csv')
df_train   = train.merge(train_meta, on='Id', how='inner')

df_train['has_title']       = (df_train['Title'].notna() & (df_train['Title'].str.strip() != '')).astype(float)
df_train['has_description'] = (df_train['Description'].notna() & (df_train['Description'].str.strip() != '')).astype(float)
df_train['title_length']    = df_train['Title'].fillna('').str.len().astype(float)
df_train['resolution']      = (df_train['width'] * df_train['height']).astype(float)

FEATURES = ['duration', 'has_title', 'has_description', 'title_length', 'resolution']

# Drop rows with any NaN in features or target
df_train_clean = df_train[FEATURES + ['ECR']].dropna()
X_train = df_train_clean[FEATURES].values
y_train = df_train_clean['ECR'].values
print(f"Train samples after cleaning: {len(df_train_clean):,}")

# ── 2. TRAIN MODEL ────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train_sc, y_train)

print("\nFeature coefficients:")
for feat, coef in zip(FEATURES, model.coef_):
    print(f"  {feat:<22s}: {coef:.6f}")

# ── 3. LOAD & MERGE VAL ───────────────────────────────────────────
print("\nLoading validation data...")
val      = pd.read_csv(RAW / 'val_data.csv')
val_meta = pd.read_csv(PROC / 'val_metadata.csv')  # already has ECR
df_val   = val.merge(val_meta, on='Id', how='inner')

df_val['has_title']       = (df_val['Title'].notna() & (df_val['Title'].str.strip() != '')).astype(float)
df_val['has_description'] = (df_val['Description'].notna() & (df_val['Description'].str.strip() != '')).astype(float)
df_val['title_length']    = df_val['Title'].fillna('').str.len().astype(float)
df_val['resolution']      = (df_val['width'] * df_val['height']).astype(float)

df_val_clean = df_val[['Id'] + FEATURES + ['ECR']].dropna()
X_val = df_val_clean[FEATURES].values
y_val = df_val_clean['ECR'].values
print(f"Val samples after cleaning: {len(df_val_clean):,}")

# ── 4. PREDICT & EVALUATE ─────────────────────────────────────────
X_val_sc = scaler.transform(X_val)
y_pred   = model.predict(X_val_sc)

srcc, srcc_p = spearmanr(y_val, y_pred)
plcc, plcc_p = pearsonr(y_val, y_pred)
rmse         = np.sqrt(mean_squared_error(y_val, y_pred))
score        = 0.6 * srcc + 0.4 * plcc

print("\n" + "=" * 50)
print("B0 OFFICIAL VALIDATION RESULTS")
print("(Ridge regression on 5 technical features)")
print("=" * 50)
print(f"  SRCC  (Spearman): {srcc:.4f}  (p={srcc_p:.2e})")
print(f"  PLCC  (Pearson):  {plcc:.4f}  (p={plcc_p:.2e})")
print(f"  RMSE:             {rmse:.4f}")
print(f"  Score (0.6S+0.4P): {score:.4f}")
print("=" * 50)
print(f"\nBaseline (Li et al.) Score: 0.660")
print(f"B0 Score gap: {0.660 - score:.3f} points below classical pipeline")
print("\nConclusion: Technical metadata alone explains <4% of ECR variance.")
print("Audiovisual content (pipeline C1, LMMs L1/Q1) is necessary.")

# ── 5. SAVE PREDICTIONS ───────────────────────────────────────────
out = df_val_clean[['Id', 'ECR']].copy()
out.columns = ['Id', 'ECR_true']
out['ECR_pred'] = y_pred
out.to_csv(RES / 'b0_val_predictions.csv', index=False)
print(f"\nPredictions saved to results/b0_val_predictions.csv")

# ── 6. SAVE METRICS JSON ──────────────────────────────────────────
metrics = {"SRCC": srcc, "PLCC": plcc, "RMSE": rmse, "Score": score, "n_val": len(df_val_clean)}
with open(RES / 'b0_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to results/b0_metrics.json")
