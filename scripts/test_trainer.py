import sys
sys.path.insert(0, ".")

from src.train.trainer import Trainer

config = {
    "arm":          "a1",
    "train_index":  "data/index_50k.parquet",
    "val_index":    "data/index_50k.parquet",
    "out_dir":      "runs/test_a1",
    "total_steps":  50,
    "warmup_steps": 5,
    "batch_size":   16,
    "lr":           3e-4,
    "lambda_aux":   0.1,
}

trainer = Trainer(config)
trainer.train()
print("Trainer test OK")