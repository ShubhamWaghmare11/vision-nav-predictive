from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.policy.idm_policy import IDMPolicy
import numpy as np
from PIL import Image
import os

os.makedirs("frames_junctions2", exist_ok=True)

env = MetaDriveEnv(dict(
    num_scenarios=400, start_seed=1000, map=4, log_level=50,
    image_observation=True, norm_pixel=False, stack_size=1,
    sensors={"rgb_camera": (RGBCamera, 84, 84)},
    vehicle_config=dict(image_source="rgb_camera"),
    traffic_density=0.0, use_render=False,
    agent_policy=IDMPolicy,  # let IDM drive so it actually reaches junctions
))

# find seeds that have X or O blocks
target_seeds = []
for seed in range(1000, 1050):
    env.reset(seed=seed)
    blocks = [b.ID for b in env.current_map.blocks]
    if "X" in blocks or "O" in blocks:
        target_seeds.append(seed)
    if len(target_seeds) == 3:
        break

print(f"Testing seeds with X/O blocks: {target_seeds}")

for seed in target_seeds:
    obs, _ = env.reset(seed=seed)
    blocks = "".join([b.ID for b in env.current_map.blocks])
    print(f"\nseed {seed}: {blocks}")
    saved = 0
    for step in range(800):
        obs, r, term, trunc, info = env.step([0.0, 0.0])  # IDM ignores action
        # save a frame every 100 steps
        if step % 100 == 0:
            frame = obs["image"][..., 0]
            Image.fromarray(frame.astype(np.uint8)).save(
                f"frames_junctions2/seed{seed}_step{step:04d}.png"
            )
            saved += 1
        if term or trunc:
            print(f"  episode ended at step {step}")
            break
    print(f"  saved {saved} frames")

env.close()
print("\nDone. Check frames_junctions2/")