import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms.functional as TF
import random
_NPZ_CACHE = {}


class DrivingDataset(Dataset):
    """
    Preloads all episode data into RAM at init time.
    Faster than per-sample disk reads on Windows.
    """

    def __init__(
        self,
        index_path: str,
        K:          int = 8,
        H:          int = 8,
        horizons:   list = [2, 4, 8],
        augment:    bool = True,
    ):
        self.df       = pd.read_parquet(index_path)
        self.K        = K
        self.H        = H
        self.horizons = horizons
        self.max_k    = max(horizons)
        self.augment  = augment

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
        row  = self.df.iloc[idx]
        t    = int(row["t"])
        data = self._episodes[row["npz_path"]]

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

        if self.augment:
            hist_t, fut_t = self._augment(hist_t, fut_t)

        return {
            "frames":        hist_t,
            "future_frames": fut_t,
            "proprio":       torch.tensor(proprios[t], dtype=torch.float32),
            "command":       torch.tensor(int(commands[t]), dtype=torch.long),
            "action_chunk":  torch.tensor(chunk, dtype=torch.float32),
        }

    def _augment(self, hist, fut):
        all_frames = torch.cat([hist, fut], dim=0).float()
        pad = 4
        all_frames = TF.pad(all_frames, pad, padding_mode='edge')
        i = random.randint(0, 2 * pad)
        j = random.randint(0, 2 * pad)
        all_frames = all_frames[:, :, i:i+84, j:j+84]
        if random.random() < 0.8:
            all_frames = TF.adjust_brightness(all_frames, 1.0 + random.uniform(-0.2, 0.2))
            all_frames = TF.adjust_contrast(all_frames,  1.0 + random.uniform(-0.2, 0.2))
            all_frames = TF.adjust_hue(all_frames, random.uniform(-0.02, 0.02))
        if random.random() < 0.1:
            all_frames = TF.rgb_to_grayscale(all_frames, num_output_channels=3)
        all_frames = all_frames.clamp(0, 255).byte()
        K = hist.size(0)
        return all_frames[:K], all_frames[K:]

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from torch.utils.data import DataLoader
    import time

    ds = DrivingDataset("data/index_50k.parquet", augment=True)
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