import sys
sys.path.insert(0, ".")
from src.train.trainer import Trainer

for arm in ["a0", "a1", "a2", "a3"]:
    print(f"\n=== Testing arm {arm} ===")
    config = {
        "arm":          arm,
        "train_index":  "data/index_50k.parquet",
        "val_index":    "data/index_50k.parquet",
        "out_dir":      f"runs/test_{arm}",
        "total_steps":  20,
        "warmup_steps": 5,
        "batch_size":   16,
        "lr":           3e-4,
        "lambda_aux":   0.1,
    }
    trainer = Trainer(config)
    trainer.train()
    print(f"arm {arm} OK")

print("\nAll arms OK")