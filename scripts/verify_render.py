from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
import numpy as np, os

os.makedirs("frames", exist_ok=True)

env = MetaDriveEnv(dict(
    num_scenarios=10, start_seed=1000, map=4, log_level=50,
    image_observation=True, norm_pixel=False, stack_size=1,
    sensors={"rgb_camera": (RGBCamera, 84, 84)},
    vehicle_config=dict(image_source="rgb_camera"),
    traffic_density=0.0, use_render=False,
))

obs, _ = env.reset(seed=1000)
img = obs["image"]
print("shape:", img.shape, "dtype:", img.dtype, "min:", img.min(), "max:", img.max())

for i in range(60):
    obs, r, term, trunc, info = env.step([0.0, 0.5])
    if i % 10 == 0:
        frame = obs["image"]
        frame = frame[...,0]
        from PIL import Image
        Image.fromarray(frame.astype(np.uint8)).save(f"frames/f{i:03d}.png")
    if term or trunc:
        obs, _ = env.reset(seed=1000)

env.close()
print("Done. Check the frames/ folder.")