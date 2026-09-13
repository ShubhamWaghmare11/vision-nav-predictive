import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import torch
import time
from pathlib import Path
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

from src.env.metadrive_wrapper import (
    TEST_CFG, VAL_CFG, build_student_obs, reset_command_state
)
from src.models.policy import VisualPolicy


class Evaluator:
    """
    Closed-loop evaluation of a trained VisualPolicy.
    Runs the policy in MetaDrive and records:
      - route completion
      - success rate
      - infractions per km
      - cross-track error
      - steering jerk
      - command compliance
    """

    def __init__(
        self,
        policy: VisualPolicy,
        device: torch.device,
        chunk_len: int = 8,
        exec_len:  int = 4,      # receding horizon: execute this many steps per chunk
        K:         int = 8,
    ):
        self.policy    = policy
        self.device    = device
        self.chunk_len = chunk_len
        self.exec_len  = exec_len
        self.K         = K
        self.policy.eval()

    def evaluate(
        self,
        env_cfg:   dict,
        seeds:     list,
        max_steps: int = 1000,
    ) -> dict:
        """
        Run one episode per seed, return aggregated metrics.
        """
        env = MetaDriveEnv(env_cfg)

        all_completion  = []
        all_success     = []
        all_infraction  = []
        all_xte         = []
        all_jerk        = []
        all_cmd_comply  = []
        all_km          = []

        for seed in seeds:
            metrics = self._run_episode(env, seed, max_steps)
            all_completion.append(metrics["completion"])
            all_success.append(float(metrics["success"]))
            all_infraction.append(metrics["infractions"])
            all_xte.append(metrics["mean_xte"])
            all_jerk.append(metrics["steering_jerk"])
            all_cmd_comply.append(metrics["cmd_compliance"])
            all_km.append(metrics["km_driven"])

        env.close()

        total_km = sum(all_km)
        total_infractions = sum(
            r * k for r, k in zip(all_infraction, all_km)
        )

        return {
            "completion_mean":   float(np.mean(all_completion)),
            "completion_std":    float(np.std(all_completion)),
            "success_rate":      float(np.mean(all_success)),
            "infractions_per_km": float(total_infractions / max(total_km, 0.001)),
            "mean_xte":          float(np.mean(all_xte)),
            "steering_jerk":     float(np.mean(all_jerk)),
            "cmd_compliance":    float(np.mean([c for c in all_cmd_comply if c >= 0])) if any(c >= 0 for c in all_cmd_comply) else -1.0,
            "n_episodes":        len(seeds),
            "seeds":             seeds,
            "per_episode": {
                "completion":  all_completion,
                "success":     all_success,
                "xte":         all_xte,
                "jerk":        all_jerk,
            }
        }

    def _run_episode(self, env, seed: int, max_steps: int) -> dict:
        raw_obs, _ = env.reset(seed=seed)
        reset_command_state(env)

        # frame buffer for K-step history
        frame_buf = []

        steers        = []
        xtes          = []
        infractions   = 0
        km_driven     = 0.0
        completion    = 0.0
        success       = False
        cmd_compliances = []

        # chunk execution state
        current_chunk = None
        chunk_pos     = 0

        prev_steer    = 0.0

        for step in range(max_steps):
            student = build_student_obs(env, raw_obs)

            # maintain frame buffer
            frame_buf.append(
                torch.from_numpy(
                    student.image.transpose(2, 0, 1).copy()
                ).unsqueeze(0)
            )
            if len(frame_buf) > self.K:
                frame_buf.pop(0)

            # pad if not enough history yet
            while len(frame_buf) < self.K:
                frame_buf.insert(0, frame_buf[0])

            # replan if needed
            if current_chunk is None or chunk_pos >= self.exec_len:
                frames_t = torch.cat(frame_buf, dim=0).unsqueeze(0).to(self.device)  # (1,K,3,84,84)
                proprio_t = torch.tensor(
                    student.proprio, dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                command_t = torch.tensor(
                    [student.command], dtype=torch.long
                ).to(self.device)

                with torch.no_grad():
                    actions, _ = self.policy(frames_t, proprio_t, command_t)
                current_chunk = actions[0].cpu().numpy()  # (H, 2)
                chunk_pos = 0

            # execute current action
            action = current_chunk[chunk_pos]
            chunk_pos += 1

            steer = float(action[0])
            steers.append(steer)

            # steering jerk
            jerk = abs(steer - prev_steer)
            prev_steer = steer

            raw_obs, _, term, trunc, info = env.step(action.tolist())

            # metrics from privileged info (eval only)
            try:
                from src.env.metadrive_wrapper import get_privileged_labels
                priv = get_privileged_labels(env)
                xtes.append(abs(priv["lat_offset"]))
            except Exception:
                xtes.append(0.0)

            # distance driven
            speed = float(env.agent.speed)
            km_driven += speed * 0.02 * 5 / 1000.0  # physics_step * decision_repeat

            # route completion
            try:
                completion = float(info.get("route_completion", 0.0))
            except Exception:
                pass

            # infractions
            if info.get("crash_vehicle", False) or info.get("crash_object", False):
                infractions += 1
            if info.get("out_of_road", False):
                infractions += 1

            # success
            if info.get("arrive_dest", False):
                success = True

            if term or trunc:
                break

        # steering jerk: mean absolute second difference
        if len(steers) >= 3:
            jerk_vals = [abs(steers[i] - 2*steers[i-1] + steers[i-2])
                        for i in range(2, len(steers))]
            steering_jerk = float(np.mean(jerk_vals))
        else:
            steering_jerk = 0.0

        return {
            "completion":    completion,
            "success":       success,
            "infractions":   infractions,
            "mean_xte":      float(np.mean(xtes)) if xtes else 0.0,
            "steering_jerk": steering_jerk,
            "cmd_compliance": -1.0,  # placeholder — junction tracking added later
            "km_driven":     km_driven,
        }


if __name__ == "__main__":
    import sys, os, json
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    from src.models.policy import VisualPolicy
    from src.env.metadrive_wrapper import TEST_CFG

    device = torch.device("cpu")
    policy = VisualPolicy().to(device)  # random weights — just testing shapes

    evaluator = Evaluator(policy, device)

    print("Running eval on 3 test seeds (random policy)...")
    results = evaluator.evaluate(
        env_cfg=TEST_CFG,
        seeds=[3000, 3001, 3002],
        max_steps=100,
    )

    print(json.dumps({k: v for k, v in results.items()
                      if k != "per_episode"}, indent=2))
    print("Evaluator OK")