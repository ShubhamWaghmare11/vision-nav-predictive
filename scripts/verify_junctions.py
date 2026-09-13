from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
import numpy as np
from PIL import Image
import os

os.makedirs("frames_junctions", exist_ok=True)

env = MetaDriveEnv(dict(
    num_scenarios=400, start_seed=1000, map=4, log_level=50,
    image_observation=True, norm_pixel=False, stack_size=1,
    sensors={"rgb_camera": (RGBCamera, 84, 84)},
    vehicle_config=dict(image_source="rgb_camera"),
    traffic_density=0.0, use_render=False,
))

junction_blocks = {"T", "X", "O", "r"}  # T-junction, intersection, roundabout, ramp
block_counts = {}
junction_maps = 0

for seed in range(1000, 1020):  # just 20 maps for a quick check
    env.reset(seed=seed)
    blocks = [b.ID for b in env.current_map.blocks]
    has_junction = any(b in junction_blocks for b in blocks)
    if has_junction:
        junction_maps += 1
    block_str = "".join(blocks)
    for b in blocks:
        block_counts[b] = block_counts.get(b, 0) + 1
    print(f"seed {seed}: {block_str}  {'<-- HAS JUNCTION' if has_junction else ''}")

    # save a frame from this map
    obs, _ = env.reset(seed=seed)
    for _ in range(30):
        obs, _, term, trunc, _ = env.step([0.0, 0.5])
        if term or trunc:
            break
    frame = obs["image"][..., 0]
    Image.fromarray(frame.astype(np.uint8)).save(f"frames_junctions/seed_{seed}.png")

env.close()

print(f"\nMaps with junctions: {junction_maps}/20")
print("Block type counts:", dict(sorted(block_counts.items())))