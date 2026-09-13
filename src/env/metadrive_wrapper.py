from dataclasses import dataclass
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.policy.idm_policy import IDMPolicy


# ── environment configs ────────────────────────────────────────────────────────

TRAIN_CFG = dict(
    num_scenarios=400, start_seed=1000, map=4, log_level=50,
    image_observation=True, norm_pixel=False, stack_size=1,
    sensors={"rgb_camera": (RGBCamera, 84, 84)},
    vehicle_config=dict(image_source="rgb_camera"),
    traffic_density=0.0, use_render=False,
    random_lane_num=False, random_lane_width=False,
    random_agent_model=False, horizon=1000,
)

VAL_CFG  = {**TRAIN_CFG, "num_scenarios": 50,  "start_seed": 2000}
TEST_CFG = {**TRAIN_CFG, "num_scenarios": 100, "start_seed": 3000}


# ── student observation ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StudentObs:
    image:   np.ndarray   # (84, 84, 3) uint8
    proprio: np.ndarray   # (5,) float32
    command: int          # 0=STRAIGHT 1=RIGHT 2=LEFT

STUDENT_FIELDS = frozenset(StudentObs.__dataclass_fields__.keys())


def build_student_obs(env, raw_obs: dict) -> StudentObs:
    """
    Build student observation from env + the raw obs dict returned by env.step/reset.
    ONLY uses: RGB image, ego speed/yaw/prev-action, derived command.
    NEVER uses: lane offset, heading error, lidar, other vehicles, map.
    """
    # image — squeeze stack dim (84,84,3,1) → (84,84,3)
    img = raw_obs["image"]
    if img.ndim == 4:
        img = img[..., 0]
    img = img.astype(np.uint8)

    # proprioception from ego vehicle only
        # proprioception from ego vehicle only
    vehicle = env.agent
    speed      = float(vehicle.speed) / 40.0
    # yaw rate: approximate from heading change
    vel        = vehicle.velocity          # (vx, vy) in world frame
    heading    = vehicle.heading_theta
    # project velocity onto lateral axis for yaw-rate proxy
    yaw_rate   = float(vel[0] * -np.sin(heading) + vel[1] * np.cos(heading)) / 20.0
    prev_steer = float(vehicle.steering)
    prev_accel = float(vehicle.throttle_brake)
    lane_w     = 1.0

    proprio = np.array([speed, yaw_rate, prev_steer, prev_accel, lane_w],
                       dtype=np.float32)

    cmd = _derive_command(env)

    return StudentObs(image=img, proprio=proprio, command=cmd)


# ── command derivation ─────────────────────────────────────────────────────────

_last_cmd = {}

def _derive_command(env) -> int:
    nav = env.agent.navigation
    if nav is None:
        return 0

    checkpoint1, checkpoint2 = nav.get_checkpoints()
    if checkpoint1 is None or checkpoint2 is None:
        return _last_cmd.get(id(env), 0)

    ego_pos     = env.agent.position[:2]
    ego_heading = env.agent.heading_theta

    target_pt = checkpoint2
    dx = target_pt[0] - ego_pos[0]
    dy = target_pt[1] - ego_pos[1]
    target_angle = np.arctan2(dy, dx)
    diff = target_angle - ego_heading
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    phi  = np.degrees(diff)

    env_id = id(env)
    prev   = _last_cmd.get(env_id, 0)

    if prev == 0:
        if phi >  15: cmd = 2
        elif phi < -15: cmd = 1
        else: cmd = 0
    else:
        if phi >  6:  cmd = 2
        elif phi < -6: cmd = 1
        else: cmd = 0

    _last_cmd[env_id] = cmd
    return cmd


def reset_command_state(env):
    """Call on episode reset to clear hysteresis."""
    _last_cmd.pop(id(env), None)


# ── privileged labels (analysis only) ────────────────────────────────────────

def get_privileged_labels(env) -> dict:
    """
    Ground-truth geometric labels for probes and RSA.
    MUST NEVER be called from training or inference code.
    """
    vehicle = env.agent
    nav     = vehicle.navigation
    lane    = nav.current_lane if nav else None

    lat_off     = 0.0
    heading_err = 0.0

    if lane is not None:
        try:
            local = lane.local_coordinates(vehicle.position)
            lat_off = float(local[1])
            lane_heading = lane.heading_theta_at(local[0])
            heading_err  = float(vehicle.heading_theta - lane_heading)
            heading_err  = (heading_err + np.pi) % (2 * np.pi) - np.pi
        except Exception:
            pass

    return {
        "lat_offset":  lat_off,
        "heading_err": heading_err,
        "speed":       float(vehicle.speed),
        "command":     _last_cmd.get(id(env), 0),
    }