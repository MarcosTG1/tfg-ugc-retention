# CONTEXT.md — Project Intellectual Map

> Last updated: 2026-03-10

---

## TL;DR

This project runs a controlled head-to-head comparison between a classical multimodal pipeline and an LMM-based pipeline for predicting short-video retention (ECR) on the SnapUGC dataset. Neither architecture is built from scratch — the contribution is in the rigorous reproduction, ablation, and cross-paradigm ensemble. The original result to beat is the ICCV VQualA 2025 winner score of 0.710.

---

## Problem Statement

Short-video platforms (Snapchat, TikTok, Instagram Reels) need to predict viewer retention at cold-start — before the recommendation system has any engagement signal. Likes and view counts are popularity-biased and unavailable for new content. The right proxy is **ECR (Engagement Continuation Rate)**: the probability that a viewer watches past the first 5 seconds. It is purely content-driven and computable on day zero.

**Why is ECR hard to predict?**
- Videos are 5–60 seconds long — too short for many temporal models to exploit structure.
- Content is extremely heterogeneous (UGC): shaky cameras, speech, music, ambient noise, text overlays, hard cuts.
- Low-level quality (blur, compression artifacts) and high-level semantics (is this emotionally engaging?) both matter, and no single model captures both.

**Formal task:** Given a video + optional metadata (title, description), predict a scalar ECR ∈ [0, 1]. Evaluation metric:

```
Score = 0.6 × SRCC + 0.4 × PLCC
```

PLCC requires logistic fitting of predictions before computing correlation. SRCC and PLCC measure rank correlation and linear correlation respectively.

---

## Dataset

**SnapUGC v2** — the official dataset of the ICCV VQualA 2025 challenge.

| Property | Value |
|---|---|
| Total videos | 120,651 |
| Duration range | 5–60 seconds |
| Content source | Snapchat Spotlight |
| Annotation depth | >2,000 real viewers per video |
| Primary label | ECR (challenge v2) |
| Secondary label | NAWP — Normalized Average Watch Percentage (v1 only) |
| Split | 106,192 train / 6,000 val / 8,459 test |

**Note on ECR in v2:** The published ECR is the *normalized ranking* of the raw ECR value across all videos — not the raw P(watch > 5s). This protects commercially sensitive engagement data while preserving rank-order semantics. Models are evaluated on ranking quality (SRCC/PLCC), so the normalization does not affect the task difficulty.

**Val/test labels:** Not in the public download. Obtained directly from Dasong Li (challenge organizer, first author of `li2024delving`). Do not redistribute.

**Pre-extracted features (train + val):** Available from the CodaLab challenge forum. Covers EfficientNetV2, ResNet-3D, DOVER, mPLUG-2, YAMNet, and CLIP-text vectors. This makes it possible to train and evaluate the classical pipeline without running the expensive extraction phase from scratch.

**Content categories:** Family, Food & Dining, Pets, Hobbies, Travel, Music, Sports, and others. Category is a key variable for error analysis (H3 hypothesis).

---

## Approach

Two SOTA families of systems, reproduced on the same hardware, evaluated on the same val split:

### Pipeline A — Classical Multimodal (EVQA)

Reference: `li2024delving` (Li et al., ECCV 2024).

Four feature streams extracted independently, then fused:

| Feature | Extractor | Dim | What it captures |
|---|---|---|---|
| `feat1` | EfficientNetV2-S | 528 | Per-frame visual appearance |
| `feat2` | Distortion network | 256 | Per-frame compression / blur artifacts |
| `feat3` | 3D-ResNet18 | 512 | Per-clip motion dynamics (16-frame windows) |
| `feat4` | mPLUG-2 | 1024 | Semantic video caption embeddings |

Text metadata (title, description, caption, music genre via YAMNet) is encoded via a frozen CLIP/Stable Diffusion text encoder. Cross-attention fuses text into the visual feature space. A transformer head outputs the final ECR scalar.

**Usage in this TFG:** Inference-only. Checkpoints from the authors' Google Drive. Fine-tuning not required — the model was trained on SnapUGC train split.

### Pipeline B — LMM-based (ECNU-SJTU winner solution)

Reference: `sun2025lmmevqa` (Sun et al., ICCV VQualA 2025).

Two sub-systems:

**VideoLLaMA2-1.7B-AV**
- Joint audio-visual-language processing via a unified encoder.
- Fine-tuned on SnapUGC with LoRA adapters.
- Output: `wa5` (weighted average over 5 quality-score tokens "1"–"5"), mapped to ECR via learned MLP (dropout → FC 2048 → ReLU → FC 1).
- Conda env: `videollama2` (Python 3.9).

**Qwen2.5-VL-7B-Instruct**
- Visual + language only (no audio by design — the model lacks an audio tower).
- Stronger vision encoder than VideoLLaMA2; higher parameter count.
- Fine-tuned with QLoRA 4-bit (required by VRAM budget on Maxwell GPUs).
- Output: next-token regression.
- Conda env: `qwenvl` (Python 3.9).

**Ensemble logic:** Predictions from sub-systems are combined via a weighted average; the combined prediction is then logistic-fitted before PLCC computation, matching the challenge evaluation protocol.

---

## Architecture Decisions

### Why not build from scratch?

The TFG's scientific value lies in the *comparison and ensemble*, not in inventing a third architecture. Both `li2024delving` and `sun2025lmmevqa` are reproducible SOTA implementations. Reproducing them rigorously — matching their reported numbers on the same dataset — is itself a research contribution. Ablating them systematically is the deeper contribution.

### Why inference-only for the classical pipeline?

The EVQA model was trained on SnapUGC v1 train split, which overlaps with v2's training data. Re-training on v2 would require re-running the expensive feature extraction stage (EfficientNetV2 + mPLUG-2 + distortion network) on 106,192 videos — infeasible under the compute budget. The pre-extracted features from the challenge forum cover only train+val, not test.

### Why VideoLLaMA2 as the primary LMM?

At the time of the challenge, VideoLLaMA2 was the only open-weights LMM with *native joint audio-visual-language processing* in a single model. Audio is expected to be informative (music genre, speech energy, background noise) — discarding it would handicap any fair comparison. Qwen2.5-VL is added as a visual-only control with a stronger vision backbone.

### Why QLoRA 4-bit for Qwen2.5-VL?

The ETSE-UV server runs Maxwell-architecture GPUs (Compute Capability 5.2). Qwen2.5-VL at 7B parameters does not fit in FP16 on available VRAM. QLoRA 4-bit quantization is the minimum intervention to make fine-tuning feasible without changing the model architecture. VideoLLaMA2 at 1.7B fits in FP16.

### Why the cross-paradigm ensemble?

Classical models are sensitive to low-level quality artifacts (blur, compression, camera shake) that LMMs tend to ignore. LMMs capture semantic engagement signals (is this cute? funny? surprising?) that defeat the classical feature extractors. These are *complementary* failure modes. If the inter-model error correlation r(errors_C1, errors_L1) < 0.70, a simple weighted average will outperform either model alone. No team in the ICCV VQualA 2025 challenge attempted this — they all ensembled within the LMM paradigm.

### Maxwell GPU constraint (CC 5.2) — non-negotiable

FlashAttention-2 requires Compute Capability ≥ 8.0 (Ampere). All attention operations must use `attn_implementation="eager"`. This significantly increases memory usage and inference time. Batch size must be tuned accordingly. Mixed precision FP16 is supported but BF16 is not (requires CC ≥ 8.0). This constraint affects every LMM fine-tuning and inference script.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8.5 (EVQA env), Python 3.9 (LMM envs), Python 3.10 (eval/ensemble) |
| Deep learning | PyTorch, transformers (HuggingFace), PEFT (LoRA/QLoRA) |
| Vision encoders | EfficientNetV2-S, 3D-ResNet18, ViT (via Qwen2.5-VL) |
| Audio | YAMNet (music genre classification), VideoLLaMA2 audio tower |
| Language | CLIP text encoder, Stable Diffusion tokenizer/encoder, mPLUG-2 captions |
| LMMs | VideoLLaMA2-1.7B-AV, Qwen2.5-VL-7B-Instruct |
| Data processing | pandas, numpy, ffmpeg (frame extraction, audio separation) |
| Metrics | scipy (SRCC, PLCC, KRCC), scikit-learn |
| Notebooks | Jupyter (EDA, error analysis) |
| Environments | conda (EVQA / videollama2 / qwenvl / tfg-ugc-retention) |
| Infra | WSL2 local dev, ETSE-UV Maxwell server (ssh: martugue@server) |
| Version control | Git + GitHub (`MarcosTG1/tfg-ugc-retention`) |

---

## Original Contributions

This TFG does not propose a new model. The original contributions are analytical and experimental:

1. **EDA of SnapUGC** — Retention curve analysis, ECR distribution by category, NAWP↔ECR correlation, outlier detection, and baseline characterization of what makes a video engaging. Not in either paper.

2. **Controlled reproduction on matched hardware** — Both pipelines run on the same ETSE-UV server, same val split, same evaluation code. This eliminates hardware and data-split confounds from the comparison. Citekeys: `li2024delving`, `sun2025lmmevqa`.

3. **Cross-paradigm comparative analysis** — Head-to-head scores (Score, SRCC, PLCC, RMSE) with per-category breakdown. Identifies where each paradigm fails.

4. **Systematic modality ablation** — 9 experiment conditions (B0, C1–C3, L1–L3, Q1–Q2) isolate the contribution of visual, audio, and text modalities independently within each paradigm. Quantifies the audio contribution (H1: ΔSRCC > 0.02 expected).

5. **Cross-paradigm ensemble (E2)** — EVQA + VideoLLaMA2 (+ optionally Qwen2.5-VL). No team in ICCV VQualA 2025 tried this. The hypothesis (H2) is that complementary failure modes make r(errors) < 0.70 and ensemble outperforms max(C1, L1).

---

## Experiment Index

| ID | Description | Models used |
|----|-------------|-------------|
| B0 | Interpretable linear baseline (duration, category, has_title, has_desc) | — |
| C1 | Full classical EVQA (visual + audio + text) | EVQA |
| C2 | Classical without text metadata | EVQA |
| C3 | Classical without audio (no YAMNet genre) | EVQA |
| L1 | VideoLLaMA2 full (audio + visual + text) | VideoLLaMA2-AV |
| L2 | VideoLLaMA2 visual + text, no audio | VideoLLaMA2 |
| L3 | VideoLLaMA2 visual only | VideoLLaMA2 |
| Q1 | Qwen2.5-VL visual + text (no audio — model limitation) | Qwen2.5-VL |
| Q2 | Qwen2.5-VL visual only | Qwen2.5-VL |
| E1 | LMM ensemble L1 + Q1 (replica of Sun et al.) | VideoLLaMA2 + Qwen2.5-VL |
| **E2** | **Cross-paradigm ensemble C1 + L1 (ORIGINAL)** | **EVQA + VideoLLaMA2** |

---

## Open Questions

1. **Does the EVQA checkpoint generalize from v1 to v2?** The model was trained on SnapUGC v1. v2 extended the dataset and changed the ECR definition to normalized ranking. If the distribution shift is large, C1's score will understate the classical paradigm's true ceiling.

2. **Is r(errors_C1, errors_L1) actually low enough?** H2 assumes complementary failures. If both models fail on the same categories (e.g., music videos where audio dominates), the ensemble will not improve over either alone. This is an empirical question — the error correlation must be measured on val before committing to E2.

3. **Does QLoRA 4-bit meaningfully degrade Qwen2.5-VL?** 4-bit quantization is a hardware-driven constraint, not a design choice. It may suppress the model's ceiling compared to what Sun et al. achieved on Ampere hardware. The Q1 score should be compared with the reported 0.710 with this caveat.

4. **Is `wa5` (weighted average of quality tokens) the right output head for fine-tuning VideoLLaMA2?** The Sun et al. solution uses this mapping from a 1–5 quality scale to ECR. It assumes a monotonic relationship between LLM quality perception and viewer retention. This may break for categories where low-quality aesthetics (e.g., raw authenticity) correlate positively with retention.

5. **How stable are SRCC/PLCC on 6,000 val samples?** Small changes in prediction ordering can shift SRCC by ±0.005. Cross-validation is not possible (no train labels for v2 without re-running feature extraction). Confidence intervals on the val metrics should be reported with bootstrap resampling.

---

## References

- `li2024delving`: Dasong Li et al., *Delving Deep into Engagement Quality Video Assessment*, ECCV 2024. Classical EVQA pipeline + SnapUGC dataset.
- `sun2025lmmevqa`: Yucheng Sun et al., *LMM-EVQA: Large Multimodal Models for Engagement Quality Video Assessment*, ICCV VQualA Workshop 2025. Challenge-winning LMM solution.
