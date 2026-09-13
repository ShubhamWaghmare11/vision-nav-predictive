import json
import numpy as np
import pandas as pd
from pathlib import Path


def build_index(data_dir: str, out_path: str):
    """
    Scan all episodes in data_dir and build a flat index of valid (episode, t) pairs.
    Saves as parquet for fast loading.
    """
    data_dir = Path(data_dir)
    records  = []

    for meta_path in sorted(data_dir.glob("*.json")):
        with open(meta_path) as f:
            meta = json.load(f)

        npz_path = data_dir / f"{meta['ep_id']}.npz"
        if not npz_path.exists():
            continue

        n = meta["n_steps"]
        chunk_h  = 8    # action chunk length
        max_k    = 8    # furthest future frame we need (for latent prediction)
        # valid sample indices: need t + chunk_h + max_k frames available
        max_t = n - chunk_h - max_k - 1

        for t in range(max_t):
            records.append({
                "ep_id":        meta["ep_id"],
                "npz_path":     str(npz_path),
                "t":            t,
                "seed":         meta["seed"],
                "dagger_round": meta["dagger_round"],
                "n_steps":      n,
            })

    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)
    print(f"Index built: {len(df)} samples from {df['ep_id'].nunique()} episodes")
    print(f"Saved to {out_path}")
    return df


def make_regime_splits(index_path: str, out_dir: str, seed: int = 42):
    """
    From the full index, create fixed 50k/150k/500k subset index files.
    Subsets are nested: 50k ⊂ 150k ⊂ 500k.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    df = pd.read_parquet(index_path)
    rng = np.random.default_rng(seed)

    # shuffle at episode level to avoid contiguous-episode bias
    ep_ids = df["ep_id"].unique()
    rng.shuffle(ep_ids)

    splits = {"50k": 50_000, "150k": 150_000, "500k": len(df)}

    prev_mask = None
    for name, n in splits.items():
        # take episodes in order until we have enough samples
        selected = []
        count = 0
        for ep in ep_ids:
            ep_rows = df[df["ep_id"] == ep]
            selected.append(ep_rows)
            count += len(ep_rows)
            if count >= n:
                break
        subset = pd.concat(selected).head(n)
        out_path = out_dir / f"index_{name}.parquet"
        subset.to_parquet(out_path, index=False)
        print(f"{name}: {len(subset)} samples from "
              f"{subset['ep_id'].nunique()} episodes → {out_path}")