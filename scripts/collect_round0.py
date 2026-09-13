import sys
sys.path.insert(0, ".")

from src.data.collector import collect_episodes
from src.data.build_index import build_index, make_regime_splits

print("=== Round 0 data collection ===")
print("Target: ~500k frames, 2 workers, train seeds 1000-1399")
print("Estimated time: 2-3 hours\n")

# worker 0
collect_episodes(
    out_dir="data/round0",
    num_episodes=400,
    seed_start=0,
    seed_range=(1000, 1399),
    dagger_round=0,
    max_steps=1000,
    worker_id=0,
)

print("\n=== Building index ===")
build_index("data/round0", "data/index_full.parquet")
make_regime_splits("data/index_full.parquet", "data/")

print("\nDone. Check data/ for index files.")