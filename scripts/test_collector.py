import sys
sys.path.insert(0, ".")

from src.data.collector import collect_episodes
from src.data.build_index import build_index, make_regime_splits
import numpy as np

print("=== Test: collect 5 episodes ===")
collect_episodes(
    out_dir="data/test_collection",
    num_episodes=5,
    seed_start=42,
    seed_range=(1000, 1399),
    dagger_round=0,
    worker_id=0,
)

print("\n=== Test: build index ===")
df = build_index("data/test_collection", "data/test_index.parquet")
print(df.head())
print("Columns:", df.columns.tolist())

print("\n=== Test: load one sample ===")
row = df.iloc[100]
data = np.load(row["npz_path"])
t    = row["t"]
img  = data["frames"][t]
act  = data["actions"][t : t+8]
priv = data["PRIVILEGED_probe_labels"][t]

print(f"frame shape: {img.shape} dtype: {img.dtype}")
print(f"action chunk shape: {act.shape}")
print(f"privileged labels: {priv}")
print(f"chunk has 8 actions: {act.shape == (8,2)}")