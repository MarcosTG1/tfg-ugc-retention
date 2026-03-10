# CONVENTIONS.md — Coding Standards

> Applies to: all code, notebooks, configs, and scripts in this repository.
> If two engineers read this independently, they must make the same decision.

---

## Language Policy

| Artifact | Language | Rationale |
|---|---|---|
| Code (variables, functions, classes) | English | International reproducibility — reviewers need to read the code |
| Inline comments | English | Comments explain code; code is English |
| Docstrings | English | Same as above |
| Commit messages | English | Conventional Commits format; tooling expects English |
| Notebook markdown cells | Spanish | This is a Spanish TFG; prose is for the university audience |
| File names | lowercase_snake_case, English | Cross-platform safe, grep-friendly |
| YAML config keys | UPPER_SNAKE_CASE, English | Mirrors the Python constants they configure |

---

## Python Naming

### Variables — `snake_case`, descriptive, English

```python
# GOOD
ecr_score = 0.72
video_features = extract_features(video_path)
val_predictions: list[float] = []
gt_ecr: np.ndarray = df["ecr"].values

# BAD — abbreviations without context, or Spanish names
e = 0.72
vf = extract_features(p)
predicciones = []
```

### Constants — `UPPER_SNAKE_CASE`

```python
SNAP_UGC_TRAIN_PATH = Path("/media/5tbraid/data/martugue/SnapUGC/train_data.csv")
MAX_VIDEO_DURATION_S = 60
ECR_THRESHOLD = 0.5
EVAL_SCORE_WEIGHTS = (0.6, 0.4)  # (SRCC weight, PLCC weight)
```

### Functions — `snake_case`, verb-noun pattern

```python
# GOOD
def extract_visual_features(video_path: Path, n_frames: int = 8) -> torch.Tensor: ...
def compute_ecr_score(logits: torch.Tensor) -> float: ...
def load_snapugc_split(split: str = "val") -> pd.DataFrame: ...

# BAD — noun-only names, no indication of what the function does
def features(path): ...
def ecr(x): ...
def data(): ...
```

### Classes — `PascalCase`

```python
class ECRPredictor:          ...
class MultimodalFusionHead:  ...
class SnapUGCDataset(Dataset): ...
```

### Private helpers — leading underscore

```python
def _normalize_ecr(raw_score: float) -> float: ...
def _logistic_fit(pred: np.ndarray, gt: np.ndarray) -> np.ndarray: ...
```

### Experiment configs — dataclass with UPPER keys

```python
@dataclass
class ExperimentConfig:
    EXP_ID: str            # "L1", "Q2", "E2" — matches the experiment table
    MODEL_TYPE: str        # "videollama2" | "qwen" | "evqa" | "ensemble"
    BATCH_SIZE: int
    LEARNING_RATE: float
    N_EPOCHS: int
    MODALITIES: list[str]  # ["visual", "audio", "text"] — always explicit subset
```

---

## File Naming

### Scripts (`src/`)

```
train_videollama2.py
run_ecr_inference.py
compute_ensemble.py
evaluate_predictions.py
extract_audio_features.py
```

Rule: `<verb>_<noun>.py`. No abbreviations. If the script is experiment-specific, prefix with the experiment ID: `run_ecr_inference_evqa.py`.

### Notebooks (`notebooks/`)

```
01_eda_snapugc.ipynb
02_ecr_distribution_analysis.ipynb
03_ablation_modalities.ipynb
04_ensemble_comparison.ipynb
05_error_analysis_by_category.ipynb
```

Rule: zero-padded numeric prefix for ordering, then descriptive name. The prefix determines reading order — use the chapter/section order from the thesis.

### Results (`results/`)

```
results/
  exp_C1_evqa_val_preds.csv          # per-experiment predictions
  exp_L1_videollama2_val_preds.csv
  exp_Q1_qwen_val_preds.csv
  exp_E1_ensemble_val_preds.csv
  exp_E2_crossparadigm_val_preds.csv
  metrics_summary.csv                # all experiments, all metrics
```

Rule: `exp_<ID>_<model_short>_<split>_preds.csv`. The `<ID>` must match the experiment table in `CONTEXT.md`. Never use date-stamped filenames like `preds_2026-03-10.csv` — the git history is the timestamp.

### Configs (`configs/`)

```
configs/
  videollama2_base.yaml
  qwen_base.yaml
  evqa_base.yaml
  ensemble_weights.yaml
```

---

## Function Signatures

**Every function must:**
1. Type-annotate all arguments and return type.
2. Have at minimum a one-line docstring.
3. Use `pathlib.Path` for file paths, not raw strings.
4. Use `torch.Tensor` for tensor types, never `Any`.

```python
from pathlib import Path
import torch

def extract_audio_features(
    video_path: Path,
    sample_rate: int = 16000,
    device: str = "cuda",
) -> torch.Tensor:
    """Extract YAMNet audio embeddings from a video file."""
    ...

def evaluate_predictions(
    pred_ecr: np.ndarray,
    gt_ecr: np.ndarray,
) -> dict[str, float]:
    """Compute SRCC, PLCC, and combined score for a set of predictions.

    Returns dict with keys: srcc, plcc, score.
    """
    ...
```

**Training loops:**
- Log loss every N steps (default N=50), not every step.
- Save checkpoint every epoch; keep only the last 2.
- Always use `tqdm` for progress bars — never bare `range()`.

**Data loaders:**
- Always set `num_workers >= 4` on GPU machines.
- Always set `pin_memory=True` when using CUDA.
- Always set a fixed `generator=torch.Generator().manual_seed(42)` for shuffle.

---

## Metrics Naming

The challenge metric is `Score = 0.6 × SRCC + 0.4 × PLCC`. PLCC requires logistic fitting of predictions before computation.

```python
# Standard variable names — use these everywhere, never abbreviate differently
srcc_score: float    # Spearman rank correlation coefficient
plcc_score: float    # Pearson linear correlation (after logistic fitting)
final_score: float   # 0.6 * srcc_score + 0.4 * plcc_score
gt_ecr: np.ndarray   # ground truth ECR values (shape: [N])
pred_ecr: np.ndarray # predicted ECR values (shape: [N])

# Always report in this order when printing/logging:
# SRCC=0.XXX | PLCC=0.XXX | Score=0.XXX
```

PLCC convention: always apply logistic fitting via `scipy.optimize.curve_fit` before computing Pearson correlation. The fitting function is:

```python
def _logistic(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d
```

---

## Git & Branches

### Branch structure

```
main            # stable, reproducible results only — no WIP commits
dev             # integration branch — merge feature branches here first
exp/ablation    # C1–C3, L1–L3, Q1–Q2 modality ablation
exp/ensemble    # E1, E2 ensemble experiments
exp/eda         # exploratory analysis, notebooks
```

Rule: never commit unfinished or non-reproducible code to `main`. `main` must always reproduce the numbers in `results/metrics_summary.csv`.

### Commit messages — Conventional Commits

```
feat(lmm): add VideoLLaMA2 fine-tuning script for L1
fix(eval): correct logistic fitting for PLCC computation
data(eda): add ECR distribution notebook with category breakdown
exp(ensemble): implement cross-paradigm E2 weighted average
docs: update CONTEXT.md with Maxwell GPU constraint rationale
chore: add nbstripout pre-commit hook
```

Format: `<type>(<scope>): <imperative description>`. Types: `feat`, `fix`, `exp`, `data`, `docs`, `chore`, `refactor`. Scope is the module or experiment ID.

### Tags

```
v0.1-baseline    # EVQA C1 reproduced, matching reported score
v0.2-lmm         # VideoLLaMA2 L1 + Qwen Q1 reproduced
v1.0-final       # all ablations + E2 ensemble complete
```

### `.gitignore` must include

```
*.pth
*.pt
*.mp4
*.avi
*.wav
data/
checkpoints/
__pycache__/
.ipynb_checkpoints/
*.egg-info/
wandb/
.env
outputs/      # model outputs go in results/ (versioned), not outputs/
```

---

## Notebook Conventions

### Cell order (mandatory)

**Cell 1 — Imports, seed, device:**
```python
import torch
import numpy as np
import pandas as pd
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
```

**Cell 2 — Constants (paths, hyperparams):**
```python
DATA_ROOT = Path("/media/5tbraid/data/martugue/SnapUGC")
RESULTS_DIR = Path("../results")
VAL_CSV = DATA_ROOT / "val_data.csv"
```

**Last cell — Save outputs:**
```python
results_df.to_csv(RESULTS_DIR / "exp_XX_val_preds.csv", index=False)
print("Saved.")
```

### Additional rules
- Markdown cells: Spanish prose.
- Code comments: English.
- Clear all outputs before committing (`nbstripout` pre-commit hook recommended).
- Every notebook must be runnable top-to-bottom without errors on a fresh kernel.

---

## Error Handling

Use explicit exceptions with informative messages. Never silent-fail.

```python
# GOOD
if not video_path.exists():
    raise FileNotFoundError(f"Video not found: {video_path}")

if split not in {"train", "val", "test"}:
    raise ValueError(f"Unknown split '{split}'. Expected one of: train, val, test.")

if pred_ecr.shape != gt_ecr.shape:
    raise ValueError(
        f"Shape mismatch: pred={pred_ecr.shape}, gt={gt_ecr.shape}"
    )

# BAD
try:
    process_video(path)
except:
    pass

try:
    load_features(path)
except Exception:
    return None  # silent failure — don't do this
```

Log errors with `print` or the `logging` module. Use `logging.warning` for recoverable issues (e.g., missing audio track), `raise` for unrecoverable ones.

---

## Anti-patterns

The following are **forbidden** in this repository.

**1. Magic numbers without constants**
```python
# FORBIDDEN
score = 0.6 * srcc + 0.4 * plcc   # where do 0.6 and 0.4 come from?

# REQUIRED
SRCC_WEIGHT, PLCC_WEIGHT = 0.6, 0.4
score = SRCC_WEIGHT * srcc + PLCC_WEIGHT * plcc
```

**2. Hardcoded absolute paths in source code**
```python
# FORBIDDEN
df = pd.read_csv("/media/5tbraid/data/martugue/SnapUGC/val_data.csv")

# REQUIRED — use config or Path constant at top of file
df = pd.read_csv(VAL_CSV)
```

**3. Bare `except` or silent exception swallowing**
```python
# FORBIDDEN
try:
    result = model(inputs)
except:
    result = 0.5  # default — nobody will notice this failure
```

**4. Mixing SRCC/PLCC computation with logistic fitting in ad-hoc ways**
```python
# FORBIDDEN — rolling your own logistic fit inline
from scipy.stats import pearsonr
plcc, _ = pearsonr(preds, gt)   # wrong: PLCC requires logistic fitting first

# REQUIRED — always use src/evaluation/metrics.py:compute_plcc()
```

**5. Committing model checkpoints or video files**
```
# FORBIDDEN — these must be in .gitignore
git add checkpoints/evqa_best.pth
git add data/sample.mp4
```

**6. Using `Any` as a type annotation**
```python
# FORBIDDEN
def run_inference(model: Any, inputs: Any) -> Any: ...

# REQUIRED — be specific
def run_inference(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor: ...
```

**7. Experiment results computed but not saved to `results/`**

Every script that produces SRCC/PLCC numbers must write them to `results/metrics_summary.csv` and the raw predictions to `results/exp_<ID>_*_preds.csv`. Numbers that exist only in a notebook cell output or a terminal log are not reproducible and do not count.
