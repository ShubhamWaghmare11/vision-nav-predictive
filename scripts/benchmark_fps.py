from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
import numpy as np
import time
import multiprocessing as mp

def worker_fps(worker_id, n_steps=500):
    env = MetaDriveEnv(dict(
        num_scenarios=400, start_seed=1000, map=4, log_level=50,
        image_observation=True, norm_pixel=False, stack_size=1,
        sensors={"rgb_camera": (RGBCamera, 84, 84)},
        vehicle_config=dict(image_source="rgb_camera"),
        traffic_density=0.0, use_render=False,
    ))
    env.reset(seed=1000 + worker_id)
    start = time.time()
    for i in range(n_steps):
        obs, r, term, trunc, info = env.step([0.0, 0.5])
        _ = obs["image"][..., 0]  # simulate the squeeze we'll do in collection
        if term or trunc:
            env.reset(seed=1000 + worker_id)
    elapsed = time.time() - start
    env.close()
    fps = n_steps / elapsed
    print(f"  worker {worker_id}: {fps:.1f} FPS", flush=True)
    return fps

if __name__ == "__main__":
    for n_workers in [1, 2, 4]:
        print(f"\n--- {n_workers} worker(s) ---")
        start = time.time()
        with mp.Pool(n_workers) as pool:
            results = pool.map(worker_fps, range(n_workers))
        total = sum(results)
        print(f"  total: {total:.1f} FPS  |  wall: {time.time()-start:.1f}s")