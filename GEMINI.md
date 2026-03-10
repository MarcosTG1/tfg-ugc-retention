# GEMINI.md — Agent Operational Guide

> For: Claude Code, Gemini CLI, or any AI coding agent working in this repo.
> Scope: `tfg-ugc-retention/` only. 

---

## Quick Start

```bash
conda activate tfg-ugc-retention  # EDA, eval, ensemble (Python 3.10)
conda activate EVQA               # Classical pipeline inference (Python 3.8.5)
conda activate videollama2        # VideoLLaMA2 fine-tune/inference (Python 3.9)
conda activate qwenvl             # Qwen2.5-VL fine-tune/inference (Python 3.9)
nvidia-smi                        # Always check GPU before training
```

---

## ⛔ DO NOT TOUCH

The following must never be modified, deleted, or overwritten — not even "accidentally":

| Category | Pattern |
|---|---|
| Model checkpoints | `*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`, `*.bin` |
| Raw video/audio | `*.mp4`, `*.avi`, `*.wav`, `*.mp3` |
| Dataset CSVs on server | `/media/5tbraid/data/martugue/SnapUGC/**` |
| Environment definitions | `environment.yml` (read unless explicitly asked) |
| Sibling directories | Anything outside `tfg-ugc-retention/` |
| Git history | Never `--force-push`, never `git reset --hard` on `main` |
| knowledge/ | PDFs and reference docs — read-only |

---

## Repository Structure

```
tfg-ugc-retention/
├── src/
│   ├── baseline/           # B0 — linear baseline (duration, category, metadata)
│   ├── classical_pipeline/ # C1–C3 — EVQA inference wrappers
│   ├── lmm_pipeline/       # L1–L3, Q1–Q2 — VideoLLaMA2 / Qwen wrappers
│   ├── ensemble/           # E1, E2 — prediction combination logic
│   └── evaluation/         # SRCC, PLCC, logistic fit, combined score
├── notebooks/              # EDA + analysis (prose in Spanish, code comments in English)
│   ├── 1 - EstadísticasDescriptivasyDistribuciones.ipynb
│   └── 2 - CorrelacionesyPredictores.ipynb
├── scripts/                # Utility scripts (metadata extraction, figure generation)
│   ├── train_metadata_extraction.py
│   ├── val_metadata_extraction.py
│   ├── evaluate_b0.py
│   └── generate_new_figures.py
├── videollama2/            # VideoLLaMA2 dataset prep + training entry points
│   ├── prepare_dataset.py
│   └── train.sh            # ⚠ Update absolute paths before running on server
├── configs/                # YAML configs per experiment (currently empty)
├── data/
│   ├── raw/                # train_data.csv, val_data.csv, test_data.csv (tracked)
│   │   ├── val_truth.csv   # val ECR ground truth (from Dasong Li — do not redistribute)
│   │   └── test_truth.csv  # test ECR ground truth
│   └── processed/          # Derived metadata CSVs (train/val/test_metadata.csv)
├── results/                # Metrics JSONs, prediction CSVs, plots (versioned)
│   ├── b0_metrics.json
│   ├── b0_val_predictions.csv
│   └── *.png               # EDA plots
├── knowledge/              # Reference PDFs very important
├── environment.yml         # tfg-ugc-retention conda env (Python 3.10)
├── CONTEXT.md              # Intellectual map — what this project is and why
├── CONVENTIONS.md          # Coding standards — naming, formatting, metrics
└── GEMINI.md               # This file
```

**Server dataset path** (not present locally):
```
/media/5tbraid/data/martugue/SnapUGC/
├── train_data.csv
├── val_data.csv
├── test_data.csv
├── train_videos/
├── val_videos/
└── val_features/    # pre-extracted features from CodaLab forum
```

---

## Conda Environments

| Env | Python | Use |
|---|---|---|
| `tfg-ugc-retention` | 3.10 | EDA notebooks, evaluation, ensemble |
| `EVQA` | 3.8.5 | C1–C3 classical pipeline inference |
| `videollama2` | 3.9 | L1–L3 VideoLLaMA2 fine-tuning / inference |
| `qwenvl` | 3.9 | Q1–Q2 Qwen2.5-VL fine-tuning / inference |

---

## Running Experiments

### Classical pipeline — EVQA (C1–C3)

```bash
conda activate EVQA
# Checkpoints must be in ECR_inference/checkpoints/ (from Google Drive)
# Required: EVQA.pth, mPLUG2_MSRVTT_Caption.pth, net_distort6_g_latest.pth, r3d18_K_200ep.pth
python test_SnapUGC_baseline.py \
  --videos_dir /media/5tbraid/data/martugue/SnapUGC/val_videos/ \
  --csv_file /media/5tbraid/data/martugue/SnapUGC/val_data.csv
# Output: submission_baseline.csv (columns: Id, ECR)
```

### VideoLLaMA2 — L1/L2/L3

```bash
conda activate videollama2
cd videollama2/

# 1. Prepare dataset JSON (run once)
python prepare_dataset.py

# 2. Fine-tune (update paths in train.sh first — all /root/workspace/... must change)
#    Required env vars in train.sh: model_path, data_path, output_dir, audio_tower,
#    pretrain_mm_mlp_adapter_a
bash train.sh

# 3. Run validation
bash run_validation.sh   # if available, else run inference script manually
```

### Qwen2.5-VL — Q1/Q2

```bash
conda activate qwenvl
cd LMM-EVQA/Qwen2.5-VL/   # path on server after rsync

# Fine-tune (QLoRA 4-bit — required for Maxwell VRAM budget)
bash train.sh

# Inference / validation
bash run_validation.sh
```

**Critical server constraint:** Maxwell GPUs (Compute Capability 5.2). FlashAttention-2 is unavailable (requires CC ≥ 8.0). Always set `attn_implementation="eager"` in all model loading calls. BF16 is unsupported — use FP16 only.

### Evaluate any experiment

```bash
conda activate tfg-ugc-retention
python scripts/evaluate_b0.py   # example; adapt for other experiments
# Always save results to results/exp_<ID>_val_preds.csv and results/b0_metrics.json
```

---

## Server Operations

### Server path reference

| Variable | Path |
|---|---|
| `HOME` | `/media/2tbraid/martugue/` |
| `data_path` | `/media/5tbraid/data/martugue/SnapUGC` |
| `output_dir` | `$HOME/TFG/models/<experiment-name>` |
| `model_path` | `$HOME/TFG/models/<downloaded-checkpoint>` |

### Sync code from local WSL to server

```bash
# Run from local WSL — never from inside the server
rsync -avz --progress \
  ~/Workspace/TFG/ \
  martugue@<server-ip>:~/TFG/code/ \
  --exclude='__pycache__' \
  --exclude='*.pth' \
  --exclude='*.mp4' \
  --exclude='.git'
```

### GPU check (always run before training)

```bash
nvidia-smi
```

### Long-running jobs — tmux is mandatory

```bash
tmux new -s tfg        # create session
# Ctrl+B, D            # detach (job keeps running)
tmux attach -t tfg     # reattach
tmux ls                # list sessions
```

---

## Experiment ID Reference

| ID | Module | Modalities | Description |
|----|--------|------------|-------------|
| B0 | `src/baseline/` | — | Linear baseline: duration, category, has_title, has_desc |
| C1 | `src/classical_pipeline/` | V+A+T | Full EVQA (Li et al., ECCV 2024) |
| C2 | `src/classical_pipeline/` | V+A | Classical without text metadata |
| C3 | `src/classical_pipeline/` | V+T | Classical without audio (no YAMNet) |
| L1 | `src/lmm_pipeline/` | V+A+T | VideoLLaMA2-1.7B-AV full |
| L2 | `src/lmm_pipeline/` | V+T | VideoLLaMA2 without audio |
| L3 | `src/lmm_pipeline/` | V | VideoLLaMA2 visual only |
| Q1 | `src/lmm_pipeline/` | V+T | Qwen2.5-VL-7B-Instruct (no audio by design) |
| Q2 | `src/lmm_pipeline/` | V | Qwen2.5-VL visual only |
| E1 | `src/ensemble/` | — | LMM+LMM ensemble: L1 + Q1 (Sun et al. replica) |
| **E2** | `src/ensemble/` | — | **Cross-paradigm ensemble: C1 + L1 (original contribution)** |

When writing commits or filenames, always use the ID from this table.

---

## Commit Format

Conventional Commits, English, 72-char subject line max.

```
<type>(<scope>): <short imperative description>

[optional body — explains WHY, not WHAT]
[reference experiment ID: B0, C1-C3, L1-L3, Q1-Q2, E1, E2]
```

**Allowed types:** `feat` | `fix` | `exp` | `data` | `docs` | `refactor` | `chore` | `wip`

**Scope examples:** `ecr-inference` | `lmm-evqa` | `videollama2` | `qwen` | `eda` | `ablation` | `ensemble` | `eval`

```bash
# GOOD
git commit -m "feat(lmm-evqa): add wa5 regression head to VideoLLaMA2 pipeline"
git commit -m "exp(ablation): run modality dropout experiment L2 on val set"
git commit -m "data(eda): add ECR distribution notebook with category breakdown"
git commit -m "fix(ecr-inference): correct ffmpeg audio extraction path on server"
git commit -m "chore: update .gitignore to exclude *.pth and val_videos/"

# BAD
git commit -m "updated stuff"
git commit -m "WIP"
git commit -m "fix bug"
```

**Never commit:** `*.pth`, `*.pt`, raw videos, dataset CSVs from the server, API keys, `.env` files.

---

## Branch Strategy

```
main          # stable only — reproducible results, matches results/metrics_summary.csv
dev           # integration branch — merge here before main
exp/ablation  # C1–C3, L1–L3, Q1–Q2 modality ablation work
exp/ensemble  # E1, E2 ensemble experiments
exp/eda       # EDA notebooks and exploratory scripts
```

---

## Metric Conventions (non-negotiable)

```python
# Primary metric
ecr ∈ [0, 1]   # Engagement Continuation Rate

# Evaluation
srcc_score: float   # Spearman rank correlation
plcc_score: float   # Pearson after logistic fitting (scipy.optimize.curve_fit)
final_score: float  # 0.6 * srcc_score + 0.4 * plcc_score

# Always report: SRCC=0.XXX | PLCC=0.XXX | Score=0.XXX
# Always save predictions to: results/exp_<ID>_<split>_preds.csv
# Always save metrics to: results/b0_metrics.json (or equivalent per experiment)
```

PLCC requires logistic fitting before Pearson computation — never compute raw Pearson on raw predictions. Use `src/evaluation/` for all metric computation.

---

## Agent Decision Rules

1. **Unsure which env to use?** Check the experiment ID → table above → use the mapped env.
2. **About to modify `environment.yml`?** Stop. Ask first.
3. **Script has hardcoded `/root/workspace/...` paths?** Update to server paths from the table above. Never update to local WSL paths.
4. **Writing a new script?** Put it in `scripts/` (utility) or `src/<module>/` (library code). Not at repo root.
5. **Producing predictions or metrics?** Save to `results/` with the `exp_<ID>_` prefix. Do not print-only.
6. **Touching notebooks?** Clear outputs before committing. Markdown cells in Spanish, code comments in English.
