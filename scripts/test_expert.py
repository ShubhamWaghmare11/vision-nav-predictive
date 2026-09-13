import sys
sys.path.insert(0, ".")

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from src.env.metadrive_wrapper import (
    TRAIN_CFG, build_student_obs, get_privileged_labels, reset_command_state
)
import numpy as np

cfg = {**TRAIN_CFG, "agent_policy": IDMPolicy}
env = MetaDriveEnv(cfg)
raw_obs, _ = env.reset(seed=1000)
reset_command_state(env)

print("=== 10 steps with IDM driving ===\n")
for i in range(10):
    raw_obs, r, term, trunc, info = env.step([0.0, 0.0])
    # v = env.agent
    # print(dir(v))
    # break
    student = build_student_obs(env, raw_obs)
    priv    = get_privileged_labels(env)

    print(f"step {i:02d} | img:{student.image.shape} dtype:{student.image.dtype} "
          f"proprio:{np.round(student.proprio, 2)} cmd:{student.command} "
          f"| lat_off:{priv['lat_offset']:.3f}m "
          f"heading_err:{priv['heading_err']:.3f}rad")

    if term or trunc:
        print("episode ended early")
        break

env.close()
print("\nDone. Verify lat_off and heading_err are NOT in student obs.")