import sys
sys.path.insert(0, ".")

from src.data.collector import collect_episodes
from src.data.build_index import build_index, make_regime_splits

print("=== Top-up collection: 500 episodes with spawn perturbation ===")
collect_episodes(
    out_dir="data/round0",
    num_episodes=500,
    seed_start=12345,
    seed_range=(1000, 1399),
    dagger_round=0,
    max_steps=1000,
    worker_id=1,
)

print("\n=== Rebuilding full index ===")
build_index("data/round0", "data/index_full.parquet")
make_regime_splits("data/index_full.parquet", "data/")
print("\nTop-up done.")