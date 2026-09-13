import sys, os, json, argparse
sys.path.insert(0, ".")

import torch
from src.models.policy import VisualPolicy
from src.eval.evaluator import Evaluator
from src.env.metadrive_wrapper import TEST_CFG, VAL_CFG

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",     required=True, help="path to checkpoint .pt file")
parser.add_argument("--split",    default="test", choices=["val", "test"])
parser.add_argument("--n_seeds",  type=int, default=100)
parser.add_argument("--max_steps",type=int, default=1000)
parser.add_argument("--out",      default=None, help="path to save results json")
args = parser.parse_args()

# load checkpoint
ckpt = torch.load(args.ckpt, map_location="cpu")
cfg  = ckpt["config"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = VisualPolicy(
    chunk_len=cfg.get("chunk_len", 8),
    K=cfg.get("K", 8),
    d_model=cfg.get("d_model", 256),
    n_layers=cfg.get("n_layers", 4),
    n_heads=cfg.get("n_heads", 4),
).to(device)
policy.load_state_dict(ckpt["policy"])
print(f"Loaded checkpoint from step {ckpt['step']}, arm={cfg.get('arm','?')}")

# seeds
if args.split == "test":
    seeds = list(range(3000, 3000 + args.n_seeds))
    env_cfg = TEST_CFG
else:
    seeds = list(range(2000, 2000 + args.n_seeds))
    env_cfg = VAL_CFG

evaluator = Evaluator(policy, device)
print(f"Evaluating on {len(seeds)} {args.split} seeds...")
results = evaluator.evaluate(env_cfg=env_cfg, seeds=seeds, max_steps=args.max_steps)

# print summary
print(json.dumps({k: v for k, v in results.items() if k != "per_episode"}, indent=2))

# save
out_path = args.out or args.ckpt.replace(".pt", f"_eval_{args.split}.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {out_path}")