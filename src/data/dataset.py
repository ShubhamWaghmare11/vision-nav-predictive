import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
_NPZ_CACHE = {}


class DrivingDataset(Dataset):
    """
    Preloads all episode data into RAM at init time.
    Faster than per-sample disk reads on Windows.

    Returns raw uint8 frames — augmentation happens batched on GPU in the
    trainer (see src/data/augmentation.py), not here.
    """

    def __init__(
        self,
        index_path: str,
        K:          int = 8,
        H:          int = 8,
        horizons:   list = [2, 4, 8],
    ):
        self.df       = pd.read_parquet(index_path)
        self.K        = K
        self.H        = H
        self.horizons = horizons
        self.max_k    = max(horizons)

        # plain numpy arrays for the columns read on every __getitem__ —
        # avoids per-call pandas.iloc overhead (Series construction).
        self._npz_paths = self.df["npz_path"].to_numpy()
        self._t         = self.df["t"].to_numpy()

        print("Preloading episodes into RAM...", flush=True)
        self._episodes = {}
        npz_paths = self.df["npz_path"].unique()
        for i, path in enumerate(npz_paths):
            data = np.load(path)
            self._episodes[path] = {
                "frames":   np.array(data["frames"]),
                "proprios": np.array(data["proprios"]),
                "commands": np.array(data["commands"]),
                "actions":  np.array(data["actions"]),
            }
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(npz_paths)} episodes loaded", flush=True)
        print("Preload complete.", flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        t    = int(self._t[idx])
        data = self._episodes[self._npz_paths[idx]]

        frames_all = data["frames"]
        proprios   = data["proprios"]
        commands   = data["commands"]
        actions    = data["actions"]
        T          = len(frames_all)

        # history frames
        hist_indices = [max(0, t - self.K + 1 + i) for i in range(self.K)]
        hist_frames  = frames_all[hist_indices]

        # future frames
        fut_indices = [min(t + k, T - 1) for k in self.horizons]
        fut_frames  = frames_all[fut_indices]

        # action chunk
        chunk_end = min(t + self.H, T)
        chunk     = actions[t:chunk_end]
        if len(chunk) < self.H:
            pad   = np.zeros((self.H - len(chunk), 2), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)

        # to tensors
        hist_t = torch.from_numpy(
            np.ascontiguousarray(hist_frames).transpose(0, 3, 1, 2)
        )
        fut_t = torch.from_numpy(
            np.ascontiguousarray(fut_frames).transpose(0, 3, 1, 2)
        )

        return {
            "frames":        hist_t,
            "future_frames": fut_t,
            "proprio":       torch.tensor(proprios[t], dtype=torch.float32),
            "command":       torch.tensor(int(commands[t]), dtype=torch.long),
            "action_chunk":  torch.tensor(chunk, dtype=torch.float32),
        }

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from torch.utils.data import DataLoader
    import time

    ds = DrivingDataset("data/index_50k.parquet")
    print(f"Dataset size: {len(ds)} samples")

    sample = ds[0]
    for k, v in sample.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i == 10:
            break
    elapsed = time.time() - t0
    print(f"\n10 batches of 32 in {elapsed:.2f}s → "
          f"{10*32/elapsed:.0f} samples/s (num_workers=0)")
    print("Dataset OK")