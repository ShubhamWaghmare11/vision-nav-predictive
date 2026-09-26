import sys
import os
import gc
import psutil

sys.path.insert(0, '.')

import torch
import numpy as np
import cv2
from pathlib import Path

from metadrive.envs.metadrive_env import MetaDriveEnv
from src.models.policy import VisualPolicy
from src.env.metadrive_wrapper import (
    TEST_CFG,
    build_student_obs,
    reset_command_state,
)

import torch.nn as nn
from torch.distributions import Normal


# =============================================================================
# CONFIG
# =============================================================================

K = 8
DEVICE = torch.device("cpu")

TEST_SEEDS = [3000, 3001, 3002, 3003, 3004]

VIDEOS_DIR = Path("videos")
VIDEOS_DIR.mkdir(exist_ok=True)

# How often to print RAM usage during an episode
RAM_LOG_EVERY = 100


# =============================================================================
# MEMORY UTILITIES
# =============================================================================

process = psutil.Process(os.getpid())


def get_ram_gb():
    """Current Python process RSS in GB."""
    return process.memory_info().rss / (1024 ** 3)


def log_ram(label):
    print(f"[RAM] {label}: {get_ram_gb():.2f} GB")


def cleanup():
    """
    Aggressive Python/PyTorch cleanup.
    Useful because MetaDrive/Panda3D can retain references.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    gc.collect()


# =============================================================================
# PPO HEAD
# =============================================================================

class PPOHead(nn.Module):
    def __init__(self, d_model=256, action_dim=2):
        super().__init__()

        self.actor_mean = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

        self.actor_log_std = nn.Parameter(
            torch.zeros(action_dim) - 0.5
        )

        self.critic = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        mean = self.actor_mean(x)

        std = self.actor_log_std.exp().expand_as(mean)

        return (
            Normal(mean, std),
            self.critic(x).squeeze(-1),
        )


# =============================================================================
# FROZEN ENCODER
# =============================================================================

class FrozenEncoder(nn.Module):
    def __init__(self, policy):
        super().__init__()

        self.encoder = policy.encoder

        for p in self.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, frames, proprio):
        B, Kf, C, H, W = frames.shape

        x = frames.reshape(
            B * Kf,
            C,
            H,
            W,
        ).float() / 255.0

        x = self.encoder(x)

        x = x.reshape(
            B,
            Kf,
            -1,
        ).mean(dim=1)

        return x


# =============================================================================
# EPISODE
# =============================================================================

def run_episode(
    encoder,
    ppo_head,
    env,
    seed,
    device,
    video_path=None,
):
    """
    Run exactly one episode.

    The environment is intentionally created outside this function and
    destroyed by the caller after the episode.
    """

    raw_obs, _ = env.reset(seed=seed)

    reset_command_state(env)

    frame_buf = []

    completion = 0.0
    success = False
    term_reason = "timeout"

    writer = None

    # -------------------------------------------------------------------------
    # Video writer
    # -------------------------------------------------------------------------

    if video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            20,
            (84, 84),
        )

        if not writer.isOpened():
            print(
                f"[WARNING] Could not open video writer: "
                f"{video_path}"
            )

            writer.release()
            writer = None

    # -------------------------------------------------------------------------
    # Episode loop
    # -------------------------------------------------------------------------

    try:

        for step in range(1000):

            student = build_student_obs(
                env,
                raw_obs,
            )

            # -------------------------------------------------------------
            # Save video frame
            # -------------------------------------------------------------

            if writer is not None:
                frame_bgr = student.image[:, :, ::-1].copy()

                writer.write(frame_bgr)

                del frame_bgr

            # -------------------------------------------------------------
            # Current frame
            # -------------------------------------------------------------

            frame = torch.from_numpy(
                student.image
                .transpose(2, 0, 1)
                .copy()
            )

            frame_buf.append(frame)

            if len(frame_buf) > K:
                old_frame = frame_buf.pop(0)
                del old_frame

            # Pad history until K frames are available
            while len(frame_buf) < K:
                frame_buf.insert(
                    0,
                    frame_buf[0],
                )

            # -------------------------------------------------------------
            # Build tensors
            # -------------------------------------------------------------

            frames_t = torch.cat(
                frame_buf,
                dim=0,
            ).unsqueeze(0).to(device)

            proprio_t = torch.tensor(
                student.proprio,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            # -------------------------------------------------------------
            # Inference
            # -------------------------------------------------------------

            with torch.inference_mode():

                latent = encoder(
                    frames_t,
                    proprio_t,
                )

                dist, _ = ppo_head(latent)

                # Deterministic evaluation
                action = dist.mean

            # -------------------------------------------------------------
            # Convert action
            # -------------------------------------------------------------

            action_np = (
                action
                .squeeze(0)
                .cpu()
                .numpy()
            )

            action_np = np.clip(
                action_np,
                -1.0,
                1.0,
            )

            # Minimum throttle
            if action_np[1] < 0.05:
                action_np[1] = 0.3

            # -------------------------------------------------------------
            # Step environment
            # -------------------------------------------------------------

            raw_obs, _, term, trunc, info = env.step(
                action_np.tolist()
            )

            completion = float(
                info.get(
                    "route_completion",
                    0.0,
                )
            )

            # -------------------------------------------------------------
            # Termination status
            # -------------------------------------------------------------

            if info.get("arrive_dest", False):
                success = True
                term_reason = "arrived"

            elif info.get("out_of_road", False):
                term_reason = "out_of_road"

            if trunc:
                term_reason = "timeout"

            # -------------------------------------------------------------
            # Memory logging
            # -------------------------------------------------------------

            if step % RAM_LOG_EVERY == 0:
                print(
                    f"      step={step:4d} | "
                    f"completion={completion:.1%} | "
                    f"RAM={get_ram_gb():.2f} GB"
                )

            # -------------------------------------------------------------
            # Episode finished
            # -------------------------------------------------------------

            if term or trunc:
                break

            # Explicitly delete per-step tensors
            del frames_t
            del proprio_t
            del latent
            del dist
            del action
            del action_np
            del frame
            del student

    finally:

        # ---------------------------------------------------------------------
        # ALWAYS release video writer
        # ---------------------------------------------------------------------

        if writer is not None:
            writer.release()

        writer = None

        # ---------------------------------------------------------------------
        # Clear frame history
        # ---------------------------------------------------------------------

        frame_buf.clear()

        # ---------------------------------------------------------------------
        # Delete local references
        # ---------------------------------------------------------------------

        cleanup()

    return (
        completion,
        success,
        term_reason,
    )


# =============================================================================
# LOAD BC ENCODER
# =============================================================================

def load_encoder(arm, device):
    """
    Load the BC checkpoint and construct the frozen visual encoder.
    """

    checkpoint_path = (
        f"checkpoints/{arm}_500k_s0_final.pt"
    )

    print(
        f"\nLoading BC checkpoint: "
        f"{checkpoint_path}"
    )

    bc_ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    bc_cfg = bc_ckpt["config"]

    bc_policy = VisualPolicy(
        chunk_len=bc_cfg.get(
            "chunk_len",
            8,
        ),
        K=bc_cfg.get(
            "K",
            8,
        ),
        d_model=bc_cfg.get(
            "d_model",
            256,
        ),
        n_layers=bc_cfg.get(
            "n_layers",
            4,
        ),
        n_heads=bc_cfg.get(
            "n_heads",
            4,
        ),
    )

    bc_policy.load_state_dict(
        bc_ckpt["policy"]
    )

    encoder = FrozenEncoder(
        bc_policy
    ).to(device)

    encoder.eval()

    # We no longer need the checkpoint dictionary or full policy object.
    del bc_ckpt
    del bc_policy

    cleanup()

    log_ram("after loading encoder")

    return encoder


# =============================================================================
# LOAD PPO HEAD
# =============================================================================

def load_ppo_head(arm, device):
    checkpoint_path = (
        f"rl_results/ppo_{arm}_final.pt"
    )

    print(
        f"Loading PPO checkpoint: "
        f"{checkpoint_path}"
    )

    ppo_ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    ppo_head = PPOHead(
        d_model=256,
        action_dim=2,
    ).to(device)

    ppo_head.load_state_dict(
        ppo_ckpt["ppo"]
    )

    ppo_head.eval()

    del ppo_ckpt

    cleanup()

    log_ram("after loading PPO head")

    return ppo_head


# =============================================================================
# CREATE ENVIRONMENT
# =============================================================================

def create_env():
    """
    Create a fresh MetaDrive environment.

    Creating one environment per seed is intentional. It prevents
    resources from previous episodes/resets from accumulating.
    """

    record_cfg = {
        **TEST_CFG,

        "num_scenarios": 1,

        # We don't need a visible MetaDrive window.
        "use_render": False,

        # Avoid loading unnecessary models ahead of time.
        "preload_models": False,
    }

    env = MetaDriveEnv(record_cfg)

    return env


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 70)
    print("MetaDrive PPO Evaluation + Video Recording")
    print("=" * 70)

    print(
        f"Initial RAM: "
        f"{get_ram_gb():.2f} GB"
    )

    # -------------------------------------------------------------------------
    # Evaluate a0 and a1 separately
    # -------------------------------------------------------------------------

    for arm in ["a0", "a1"]:

        print("\n")
        print("=" * 70)
        print(f"Recording {arm} PPO policy")
        print("=" * 70)

        # ---------------------------------------------------------------------
        # Load models
        # ---------------------------------------------------------------------

        encoder = load_encoder(
            arm,
            DEVICE,
        )

        ppo_head = load_ppo_head(
            arm,
            DEVICE,
        )

        print(
            "Loaded encoder + PPO head"
        )

        log_ram(
            f"{arm} before environments"
        )

        # ---------------------------------------------------------------------
        # Run each seed with a FRESH environment
        # ---------------------------------------------------------------------

        for seed in TEST_SEEDS:

            print("\n" + "-" * 70)

            print(
                f"{arm} | seed={seed}"
            )

            log_ram(
                "before creating MetaDrive"
            )

            env = None

            try:

                # -------------------------------------------------------------
                # Fresh MetaDrive environment
                # -------------------------------------------------------------

                env = create_env()

                log_ram(
                    "after creating MetaDrive"
                )

                # -------------------------------------------------------------
                # Video
                # -------------------------------------------------------------

                video_path = (
                    VIDEOS_DIR
                    / f"{arm}_seed{seed}.mp4"
                )

                # -------------------------------------------------------------
                # Run
                # -------------------------------------------------------------

                completion, success, reason = run_episode(
                    encoder=encoder,
                    ppo_head=ppo_head,
                    env=env,
                    seed=seed,
                    device=DEVICE,
                    video_path=video_path,
                )

                print(
                    f"\n  seed={seed} | "
                    f"completion={completion:.1%} | "
                    f"{reason} | "
                    f"{'SUCCESS' if success else 'FAILED'} | "
                    f"saved: {video_path.name}"
                )

                log_ram(
                    "after episode"
                )

            except Exception as e:

                print(
                    f"\n[ERROR] {arm} seed={seed}"
                )

                print(
                    f"{type(e).__name__}: {e}"
                )

                raise

            finally:

                # -------------------------------------------------------------
                # IMPORTANT:
                # Completely destroy MetaDrive environment before next seed.
                # -------------------------------------------------------------

                if env is not None:

                    print(
                        "Closing MetaDrive..."
                    )

                    try:
                        env.close()
                    except Exception as e:
                        print(
                            f"[WARNING] env.close() failed: {e}"
                        )

                    del env
                    env = None

                cleanup()

                log_ram(
                    "after MetaDrive cleanup"
                )

        # ---------------------------------------------------------------------
        # Destroy models before moving to next arm
        # ---------------------------------------------------------------------

        print("\nCleaning up models...")

        del encoder
        del ppo_head

        encoder = None
        ppo_head = None

        cleanup()

        log_ram(
            f"after completely cleaning {arm}"
        )

    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ALL VIDEOS SAVED")
    print("=" * 70)

    print(
        f"Directory: {VIDEOS_DIR.resolve()}"
    )

    log_ram("final")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()