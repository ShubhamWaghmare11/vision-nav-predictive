# Predictive Visual Representations for Generalizable Vision-Based Navigation

A controlled study on whether action-conditioned future latent prediction improves visual representations learned during behaviour cloning for autonomous driving, and whether those representations transfer to RL fine-tuning.

## Paper

**Predictive Visual Representations for Generalizable Vision-Based Navigation**
Shubham Waghmare, 2026

Published on Zenodo: https://doi.org/10.5281/zenodo.23009957



## Research Question

Does adding an action-conditioned future latent prediction auxiliary objective during BC training produce better visual encoders — as measured by (1) generalisation to unseen road layouts and (2) sample efficiency in downstream PPO fine-tuning?

## Experimental Design

Four training arms, everything else held constant:

| Arm | Objective |
|---|---|
| A0 | Pure BC baseline |
| A1 | BC + action-conditioned future latent prediction (EMA target encoder, cosine loss) |
| A2 | BC + same-timestep cross-view invariance (non-predictive control) |
| A3 | BC + action-unconditioned future latent prediction |

**Simulator:** MetaDrive (procedurally generated road maps, no traffic)  
**Expert:** IDM policy  
**Policy:** 84×84 RGB → CNN encoder (1.47M params) → Transformer trunk (4L, 4H, d=256) → 8-step action chunk  
**Data regimes:** 50k, 150k, 500k demonstration steps  
**Seeds:** 3 per condition (18 headline runs + 6 ablation runs = 24 total)  
**Train maps:** seeds 1000–1399 | **Val:** 2000–2049 | **Test:** 3000–3099  

## Key Results

### BC Generalisation (50 unseen test maps)

| Arm | 50k | 150k | 500k |
|---|---|---|---|
| A0 (Pure BC) | 37.0% | 40.1% | 35.5% |
| A1 (Predictive) | **49.9%** | 34.2% | **47.3%** |
| A2 (Cross-view) | — | — | 26.0% |
| A3 (Unconditioned) | — | — | 41.7% |

Ablation ladder at 500k: A2 (26.0%) < A0 (35.5%) < A3 (41.7%) < A1 (47.3%)

### PPO RL Fine-tuning (frozen encoder, 2M steps)

| Metric | A0 | A1 |
|---|---|---|
| Final completion (last 500k steps) | 39.2% | **44.0%** |
| Floor rate (% steps below 5 km/h) | 3.5% | **0.9%** |
| Stall rate (replay diagnostic) | 45% | **22%** |

### Representation Analysis

| Metric | A0 | A1 |
|---|---|---|
| Effective rank | 54 | **81** |
| Geometry preservation (Spearman ρ) | 0.51 | **0.61** |
| Command accuracy (linear probe) | 70.7% | **84.1%** |
| Lat offset R² (IDM probe) | 0.65 | **0.70** |
| CKA between A0 and A1 | 0.088 | — |

## Repository Structure

```
src/
  models/       — VisualPolicy, CNN encoder, Transformer trunk, auxiliary heads
  env/          — MetaDrive wrapper, student observation builder
  train/        — Trainer (BC + auxiliary losses)
  eval/         — Evaluator (closed-loop)
scripts/
  batch_eval.py                    — run all 24 checkpoints on test seeds
  collect_features.py              — extract encoder features for analysis
  representation_analysis.py       — linear probes, CKA, RSA, effective rank
  deep_representation_analysis.py  — layer-wise CKA, t-SNE, geometry analysis
  record_ppo.py                    — visualise PPO policies
results/                           — BC eval JSON files (24 runs × 50 seeds)
rl_results/                        — PPO metrics and final checkpoints
representation/                    — feature vectors, labels, analysis plots
notebooks/                         — Colab training notebook
HYPOTHESES.md                      — pre-registered hypotheses
```

## Checkpoints

BC checkpoints (24 runs, ~56MB each) and encoder files are stored on Google Drive due to size. Contact for access.

PPO final checkpoints (A0 + A1) are in `rl_results/` (~0.3MB each).

## Setup

```bash
git clone https://github.com/ShubhamWaghmare11/vision-nav-predictive.git
cd vision-nav-predictive
pip install metadrive-simulator==0.4.3 torch torchvision
```

## Related Work

This project is a companion to [LLM Alignment Study](https://github.com/ShubhamWaghmare11/llm-alignment-study) — a longitudinal CKA/RSA analysis of representation evolution across pretraining → SFT → DPO in a 334M parameter language model. Together they study how training objectives shape internal representations across two very different domains.


