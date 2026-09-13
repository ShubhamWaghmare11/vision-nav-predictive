# ============================================================
# CELL 1 — Check GPU
# ============================================================
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")


# ============================================================
# CELL 2 — Mount Drive and copy dataset
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import os, shutil, time

DRIVE_DATA = "/content/drive/MyDrive/vision-nav/data"
LOCAL_DATA = "/content/data"

os.makedirs(LOCAL_DATA, exist_ok=True)

print("Copying dataset to local disk (faster I/O during training)...")
t0 = time.time()

# copy index files
for f in ["index_50k.parquet", "index_150k.parquet",
          "index_500k.parquet", "index_full.parquet"]:
    src = f"{DRIVE_DATA}/{f}"
    dst = f"{LOCAL_DATA}/{f}"
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  copied {f}")

# copy episode npz files
src_round0 = f"{DRIVE_DATA}/round0"
dst_round0 = f"{LOCAL_DATA}/round0"
if not os.path.exists(dst_round0):
    print("Copying round0 episodes (this takes a few minutes)...")
    shutil.copytree(src_round0, dst_round0)
    print(f"Done in {time.time()-t0:.0f}s")
else:
    print("round0 already on local disk")

print(f"Total copy time: {time.time()-t0:.0f}s")


# ============================================================
# CELL 3 — Install dependencies
# ============================================================
# Run this cell once per session
import subprocess
subprocess.run(["pip", "install", "metadrive-simulator", "pandas",
                "pyarrow", "wandb", "-q"], check=True)
print("Dependencies installed")


# ============================================================
# CELL 4 — Clone or upload repo
# ============================================================
# Option A: clone from GitHub (recommended)
# !git clone https://github.com/YOUR_USERNAME/vision-nav-predictive.git
# %cd vision-nav-predictive

# Option B: upload zip
# from google.colab import files
# files.upload()  # upload vision-nav-predictive.zip
# !unzip vision-nav-predictive.zip
# %cd vision-nav-predictive

import sys
sys.path.insert(0, "/content/vision-nav-predictive")
print("Repo ready")


# ============================================================
# CELL 5 — Verify setup
# ============================================================
from src.models.policy import VisualPolicy
from src.data.dataset import DrivingDataset
import torch

policy = VisualPolicy()
n = sum(p.numel() for p in policy.parameters())
print(f"Policy params: {n:,}")

ds = DrivingDataset("/content/data/index_50k.parquet", augment=False)
print(f"50k dataset: {len(ds)} samples")
print("Setup verified")


# ============================================================
# CELL 6 — Training launcher
# Run this cell once per arm/regime combination
# ============================================================
import sys, os
sys.path.insert(0, "/content/vision-nav-predictive")
from src.train.trainer import Trainer

# ── CHANGE THESE FOR EACH RUN ────────────────────────────────
ARM    = "a0"       # a0 / a1 / a2 / a3
REGIME = "50k"      # 50k / 150k / 500k
SEED   = 0          # 0 / 1 / 2
# ─────────────────────────────────────────────────────────────

DATA_DIR = "/content/data"
RUN_NAME = f"{ARM}_{REGIME}_s{SEED}"
OUT_DIR  = f"/content/drive/MyDrive/vision-nav/runs/{RUN_NAME}"

config = {
    "arm":          ARM,
    "regime":       REGIME,
    "seed":         SEED,
    "run_name":     RUN_NAME,

    "train_index":  f"{DATA_DIR}/index_{REGIME}.parquet",
    "val_index":    f"{DATA_DIR}/index_50k.parquet",  # always validate on 50k
    "out_dir":      OUT_DIR,

    "total_steps":  60_000,
    "warmup_steps": 2_000,
    "batch_size":   256,
    "lr":           3e-4,
    "weight_decay": 0.05,
    "lambda_aux":   0.1,
    "ema_tau":      0.99,

    "chunk_len":    8,
    "K":            8,
    "d_model":      256,
    "n_layers":     4,
    "n_heads":      4,
    "dropout":      0.1,

    "device":       "cuda",
}

# set seeds for reproducibility
import torch, numpy as np, random
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"Starting run: {RUN_NAME}")
trainer = Trainer(config)
trainer.train()
print(f"Run complete: {RUN_NAME}")


# ============================================================
# CELL 7 — Resume interrupted run
# ============================================================
import torch
from src.train.trainer import Trainer
from pathlib import Path

RUN_NAME = "a1_150k_s0"   # change to the run you want to resume
OUT_DIR  = f"/content/drive/MyDrive/vision-nav/runs/{RUN_NAME}"

# find latest checkpoint
ckpts = sorted(Path(OUT_DIR).glob("ckpt_*.pt"))
if not ckpts:
    print("No checkpoints found")
else:
    latest = ckpts[-1]
    print(f"Resuming from {latest.name}")

    ckpt   = torch.load(latest, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    config["train_index"] = config["train_index"].replace(
        "/content/data", "/content/data"
    )

    trainer = Trainer(config)
    trainer.policy.load_state_dict(ckpt["policy"])
    trainer.optimizer.load_state_dict(ckpt["optimizer"])
    trainer.scheduler.load_state_dict(ckpt["scheduler"])
    trainer.step = ckpt["step"]
    print(f"Resuming from step {trainer.step}")
    trainer.train()


# ============================================================
# CELL 8 — Run order (required 24 runs)
# ============================================================
# Copy this as a reference. Run one ARM/REGIME/SEED at a time.
# Each run ~45-85 min on L4/T4.
#
# REQUIRED (18 runs) — headline H1/H2:
# ARM=a0  REGIME=50k   SEED=0,1,2
# ARM=a0  REGIME=150k  SEED=0,1,2
# ARM=a0  REGIME=500k  SEED=0,1,2
# ARM=a1  REGIME=50k   SEED=0,1,2
# ARM=a1  REGIME=150k  SEED=0,1,2
# ARM=a1  REGIME=500k  SEED=0,1,2
#
# REQUIRED (6 runs) — specificity H3
# (run after headline to pick the best regime):
# ARM=a2  REGIME=<best>  SEED=0,1,2
# ARM=a3  REGIME=<best>  SEED=0,1,2
#
# RECOMMENDED (8 runs) — chunking ablation:
# ARM=a0  REGIME=150k  chunk_len=1  SEED=0,1
# ARM=a1  REGIME=150k  chunk_len=1  SEED=0,1
# ARM=a0  REGIME=150k  chunk_len=4  SEED=0,1
# ARM=a1  REGIME=150k  chunk_len=4  SEED=0,1


# ============================================================
# CELL 9 — Quick eval after a run
# ============================================================
import sys, json, torch
sys.path.insert(0, "/content/vision-nav-predictive")

from src.models.policy import VisualPolicy
from src.eval.evaluator import Evaluator
from src.env.metadrive_wrapper import TEST_CFG

RUN_NAME = "a0_50k_s0"
OUT_DIR  = f"/content/drive/MyDrive/vision-nav/runs/{RUN_NAME}"

ckpt   = torch.load(f"{OUT_DIR}/ckpt_060000_final.pt", map_location="cpu", weights_only=False)
cfg    = ckpt["config"]
device = torch.device("cuda")

policy = VisualPolicy(**{k: cfg[k] for k in
    ["chunk_len","K","d_model","n_layers","n_heads","dropout"]}).to(device)
policy.load_state_dict(ckpt["policy"])

evaluator = Evaluator(policy, device)
results   = evaluator.evaluate(
    env_cfg=TEST_CFG,
    seeds=list(range(3000, 3050)),   # 50 seeds for quick check
    max_steps=1000,
)
print(json.dumps(
    {k: v for k, v in results.items() if k != "per_episode"},
    indent=2
))

# save to drive
with open(f"{OUT_DIR}/eval_test_quick.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved.")