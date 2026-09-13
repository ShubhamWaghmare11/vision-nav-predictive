import os
import json
import time
import numpy as np
from pathlib import Path
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from src.env.metadrive_wrapper import (
    TRAIN_CFG, build_student_obs, get_privileged_labels, reset_command_state
)


def collect_episodes(
    out_dir: str,
    num_episodes: int,
    seed_start: int,
    seed_range: tuple,        # (start, end) inclusive
    dagger_round: int = 0,
    max_steps: int = 1000,
    worker_id: int = 0,
):
    """
    Collect episodes with IDM driving.
    Saves one .npz per episode into out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {**TRAIN_CFG, "agent_policy": IDMPolicy}
    env = MetaDriveEnv(cfg)

    rng = np.random.default_rng(seed_start + worker_id)
    seeds = rng.integers(seed_range[0], seed_range[1] + 1, size=num_episodes)

    total_frames = 0
    t0 = time.time()

    for ep_idx, seed in enumerate(seeds):
        seed = int(seed)
        raw_obs, _ = env.reset(seed=seed)
        reset_command_state(env)

        # spawn perturbation for recovery demonstrations
        try:
            import math
            vehicle = env.agent
            heading_perturb = float(np.random.uniform(-8, 8)) * math.pi / 180.0
            vehicle.set_heading_theta(vehicle.heading_theta + heading_perturb)
        except Exception:
            pass

        # per-episode buffers
        frames   = []
        proprios = []
        commands = []
        actions  = []
        privs    = []
        valid    = []

        for step in range(max_steps):
            student = build_student_obs(env, raw_obs)
            priv    = get_privileged_labels(env)

            # get IDM action BEFORE stepping
            policy  = env.engine.get_policy(env.agent.name)
            action  = np.array(policy.act(env.agent.name), dtype=np.float32)

            frames.append(student.image)
            proprios.append(student.proprio)
            commands.append(student.command)
            actions.append(action)
            privs.append([
                priv["lat_offset"],
                priv["heading_err"],
                priv["speed"],
                float(priv["command"]),
            ])
            valid.append(True)

            raw_obs, _, term, trunc, _ = env.step([0.0, 0.0])

            if term or trunc:
                break

        T = len(frames)
        if T < 10:
            continue   # skip very short episodes

        # save episode
        ep_id = f"r{dagger_round}_w{worker_id}_ep{ep_idx:05d}_s{seed}"
        np.savez_compressed(
            out_dir / f"{ep_id}.npz",
            frames   = np.stack(frames).astype(np.uint8),      # (T,84,84,3)
            proprios = np.stack(proprios).astype(np.float32),  # (T,5)
            commands = np.array(commands, dtype=np.int8),      # (T,)
            actions  = np.stack(actions).astype(np.float32),   # (T,2)
            PRIVILEGED_probe_labels = np.array(privs, dtype=np.float32),  # (T,4)
            valid    = np.ones(T, dtype=bool),
        )

        # save metadata sidecar
        meta = {
            "ep_id": ep_id,
            "seed": seed,
            "dagger_round": dagger_round,
            "worker_id": worker_id,
            "n_steps": T,
        }
        with open(out_dir / f"{ep_id}.json", "w") as f:
            json.dump(meta, f)

        total_frames += T
        elapsed = time.time() - t0
        fps = total_frames / elapsed
        print(f"[w{worker_id}] ep {ep_idx+1}/{num_episodes} "
              f"seed={seed} steps={T} | "
              f"total={total_frames} frames @ {fps:.1f} FPS", flush=True)

    env.close()
    print(f"[w{worker_id}] Done. {total_frames} frames in {time.time()-t0:.1f}s")
    return total_frames