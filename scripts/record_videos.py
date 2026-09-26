"""
Record top-down driving videos for trained policies on unseen test maps.

Loads every checkpoint in ../runs-to-test, identifies its (arm, seed) from
the config saved inside the checkpoint itself (not the messy filename),
and rolls each policy out on MetaDrive test seeds 3000-3005 (unseen during
training), saving one MP4 per checkpoint that plays through all seeds
back-to-back.

Usage:
    python scripts/record_videos.py
"""
import sys, os, re, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import numpy as np
import torch
import cv2

from metadrive.envs.metadrive_env import MetaDriveEnv
from src.env.metadrive_wrapper import TEST_CFG, build_student_obs, reset_command_state
from src.models.policy import VisualPolicy

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR  = REPO_ROOT.parent / "runs-to-test"
VIDEO_DIR = REPO_ROOT / "videos"

TEST_SEEDS = [3000, 3001, 3002, 3003, 3004, 3005]
MAX_STEPS_PER_EPISODE = 500
SCREEN_SIZE = (500, 500)
FPS = 15
EXEC_LEN = 4  # receding-horizon execution length, matches Evaluator default


def label_from_config(cfg: dict) -> str:
    """Derive a clean '<arm>_<seed>' label from the checkpoint's own saved
    config (out_dir), since the checkpoint filenames on disk are inconsistent
    / mistyped (e.g. 'ao_50k_s2', 'a1_50k_so')."""
    arm = cfg.get("arm", "unknown")
    out_dir = cfg.get("out_dir", "")
    m = re.search(r"_s(\d+)$", Path(out_dir).name)
    seed = f"s{m.group(1)}" if m else "s?"
    return f"{arm}_{seed}"


def load_policy(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    policy = VisualPolicy(
        chunk_len=cfg.get("chunk_len", 8),
        K=cfg.get("K", 8),
        d_model=cfg.get("d_model", 256),
        n_layers=cfg.get("n_layers", 4),
        n_heads=cfg.get("n_heads", 4),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy, cfg, label_from_config(cfg)


def rollout_and_record(env, policy, device, seed, K, exec_len, writer, label):
    raw_obs, _ = env.reset(seed=seed)
    reset_command_state(env)

    frame_buf = []
    current_chunk = None
    chunk_pos = 0

    for step in range(MAX_STEPS_PER_EPISODE):
        student = build_student_obs(env, raw_obs)

        frame_buf.append(
            torch.from_numpy(student.image.transpose(2, 0, 1).copy()).unsqueeze(0)
        )
        if len(frame_buf) > K:
            frame_buf.pop(0)
        while len(frame_buf) < K:
            frame_buf.insert(0, frame_buf[0])

        if current_chunk is None or chunk_pos >= exec_len:
            frames_t = torch.cat(frame_buf, dim=0).unsqueeze(0).to(device)
            proprio_t = torch.tensor(student.proprio, dtype=torch.float32).unsqueeze(0).to(device)
            command_t = torch.tensor([student.command], dtype=torch.long).to(device)
            with torch.no_grad():
                actions, _ = policy(frames_t, proprio_t, command_t)
            current_chunk = actions[0].cpu().numpy()
            chunk_pos = 0

        action = current_chunk[chunk_pos]
        chunk_pos += 1

        raw_obs, _, term, trunc, info = env.step(action.tolist())

        # top-down RGB frame for the video
        frame = env.render(mode="topdown", screen_size=SCREEN_SIZE, window=False)
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.putText(frame, f"{label} | seed={seed} | step={step}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)

        if term or trunc:
            break

    outcome = "arrived" if info.get("arrive_dest", False) else (
              "crashed" if (info.get("crash_vehicle") or info.get("crash_object")) else
              "out_of_road" if info.get("out_of_road", False) else "timeout")
    return outcome, step + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default=str(CKPT_DIR))
    parser.add_argument("--out_dir",  default=str(VIDEO_DIR))
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_files = sorted(ckpt_dir.glob("*.pt"))
    if not ckpt_files:
        print(f"No checkpoints found in {ckpt_dir}")
        return

    env = MetaDriveEnv(TEST_CFG)

    try:
        for ckpt_path in ckpt_files:
            policy, cfg, label = load_policy(ckpt_path, device)
            K = cfg.get("K", 8)
            video_path = out_dir / f"{label}.mp4"
            print(f"\n=== {label}  ({ckpt_path.name}) -> {video_path.name} ===")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, FPS, SCREEN_SIZE)

            for seed in TEST_SEEDS:
                outcome, n_steps = rollout_and_record(
                    env, policy, device, seed, K, EXEC_LEN, writer, label
                )
                print(f"  seed {seed}: {outcome} after {n_steps} steps")

            writer.release()
            print(f"  saved {video_path}")
    finally:
        env.close()

    print("\nAll videos recorded.")


if __name__ == "__main__":
    main()
