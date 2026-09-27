import sys, os
sys.path.insert(0, '/root/vision-nav-predictive')
# DISPLAY removed: headless EGL (p3headlessgl) is used for GPU offscreen rendering
os.environ.pop('DISPLAY', None)
os.environ.pop('LIBGL_ALWAYS_SOFTWARE', None)

import json, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.distributions import Normal
from metadrive.envs.metadrive_env import MetaDriveEnv
from src.models.policy import VisualPolicy
from src.env.metadrive_wrapper import TRAIN_CFG, build_student_obs, reset_command_state

# ── config ─────────────────────────────────────────────────────────────────────
ENCODER_DIR   = Path('/root/encoders')
OUT_DIR       = Path('/root/rl_runs')
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOTAL_STEPS   = 2_000_000
ROLLOUT_LEN   = 512
N_EPOCHS      = 4
MINIBATCH     = 64
LR            = 3e-4
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_EPS      = 0.2
ENT_COEF      = 0.01
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5
LOG_INTERVAL  = 10
SAVE_INTERVAL = 100
K             = 8
PROGRESS_SCALE  = 100.0   # reward per unit route_completion
STALL_STEPS     = 30      # end episode if slower than STALL_SPEED_KMH this many steps in a row
STALL_SPEED_KMH = 1.0
STALL_PENALTY   = 0.02    # per step below STALL_SPEED_KMH; full 100-step stall ~= -2 (worse than off-road)
STALL_TERMINAL  = 3.0     # one-time penalty when the stall cutoff ends the episode (stall ~= -2.7 discounted vs -1 off-road)
MIN_SPEED_KMH   = 5.0     # below this speed, braking is disabled and throttle is floored at MIN_THROTTLE
MIN_THROTTLE    = 0.2     # (no traffic, so full stops are never needed; removes brake-to-halt confound)
THROTTLE_INIT   = 0.3     # initial mean throttle (action[1]) so early exploration moves the car

ENV_CFG = {**TRAIN_CFG}


# ── reward ──────────────────────────────────────────────────────────────────────
def compute_reward(info, prev_completion, speed_kmh):
    completion = float(info.get('route_completion', 0.0))
    reward     = (completion - prev_completion) * PROGRESS_SCALE
    if speed_kmh < STALL_SPEED_KMH:       reward -= STALL_PENALTY
    if info.get('out_of_road', False):    reward -= 1.0
    if info.get('crash_vehicle', False):  reward -= 1.0
    if info.get('crash_object', False):   reward -= 1.0
    if info.get('arrive_dest', False):    reward += 5.0
    return reward, completion


# ── PPO head ────────────────────────────────────────────────────────────────────
class PPOHead(nn.Module):
    def __init__(self, d_model=256, action_dim=2):
        super().__init__()
        self.actor_mean    = nn.Sequential(
            nn.Linear(d_model, 128), nn.Tanh(),
            nn.Linear(128, action_dim), nn.Tanh(),
        )
        with torch.no_grad():
            self.actor_mean[2].bias[1] = float(np.arctanh(THROTTLE_INIT))
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)
        self.critic        = nn.Sequential(
            nn.Linear(d_model, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        mean  = self.actor_mean(x)
        std   = self.actor_log_std.exp().expand_as(mean)
        dist  = Normal(mean, std)
        value = self.critic(x).squeeze(-1)
        return dist, value

    def get_value(self, x):
        return self.critic(x).squeeze(-1)


# ── frozen encoder ──────────────────────────────────────────────────────────────
class FrozenEncoder(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.encoder = policy.encoder
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, frames, proprio):
        B, Kf, C, H, W = frames.shape
        x = frames.reshape(B * Kf, C, H, W).float() / 255.0
        x = self.encoder(x)
        x = x.reshape(B, Kf, -1).mean(dim=1)
        return x


# ── obs helper ──────────────────────────────────────────────────────────────────
def get_obs_tensor(env, raw_obs, frame_buf, device):
    student = build_student_obs(env, raw_obs)
    frame   = torch.from_numpy(student.image.transpose(2, 0, 1).copy()).unsqueeze(0)
    frame_buf.append(frame)
    if len(frame_buf) > K:    frame_buf.pop(0)
    while len(frame_buf) < K: frame_buf.insert(0, frame_buf[0])
    frames_t  = torch.cat(frame_buf, dim=0).unsqueeze(0).to(device)
    proprio_t = torch.tensor(student.proprio, dtype=torch.float32).unsqueeze(0).to(device)
    return frames_t, proprio_t


# ── train ───────────────────────────────────────────────────────────────────────
RESTART_EXIT_CODE = 3


def train(arm, total_steps=TOTAL_STEPS, out_root=OUT_DIR, restart_every=0):
    print('=' * 60)
    print('PPO — encoder: ' + arm + '_500k | device: ' + str(DEVICE))
    print('=' * 60)

    out_dir = Path(out_root) / ('ppo_' + arm)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt   = torch.load(ENCODER_DIR / (arm + '_500k_s0_final.pt'),
                        map_location='cpu', weights_only=False)
    cfg    = ckpt['config']
    policy = VisualPolicy(
        chunk_len=cfg.get('chunk_len', 8), K=cfg.get('K', 8),
        d_model=cfg.get('d_model', 256),   n_layers=cfg.get('n_layers', 4),
        n_heads=cfg.get('n_heads', 4),
    )
    policy.load_state_dict(ckpt['policy'])
    encoder = FrozenEncoder(policy).to(DEVICE)
    d_model = cfg.get('d_model', 256)
    print('Encoder loaded and frozen — d_model=' + str(d_model))

    ppo       = PPOHead(d_model=d_model).to(DEVICE)
    optimizer = optim.Adam(ppo.parameters(), lr=LR)
    print('PPO head params: ' + str(round(sum(p.numel() for p in ppo.parameters()) / 1e3, 1)) + 'K')

    # resume if checkpoint exists
    resume_ckpts = sorted(out_dir.glob('ppo_' + arm + '_update*.pt'),
                          key=lambda p: int(p.stem.rsplit('update', 1)[1]))
    global_step, update, metrics = 0, 0, []
    resumed_ep_count, resumed_stall_eps = 0, 0
    if resume_ckpts:
        latest = resume_ckpts[-1]
        rc = torch.load(latest, map_location='cpu', weights_only=False)
        ppo.load_state_dict(rc['ppo'])
        optimizer.load_state_dict(rc['optimizer'])
        global_step = rc['step']
        update      = rc['update']
        metrics     = rc.get('metrics', [])
        resumed_ep_count  = rc.get('ep_count', 0)
        resumed_stall_eps = rc.get('stall_eps', 0)
        print('Resumed from ' + latest.name + ' (step ' + str(global_step) + ')')

    env = MetaDriveEnv(ENV_CFG)
    raw_obs, _ = env.reset(seed=random.randint(1000, 1399))
    reset_command_state(env)

    frame_buf       = []
    prev_completion = 0.0
    stall_count     = 0
    ep_floor_steps  = 0
    ep_reward       = 0.0
    ep_len          = 0
    ep_count        = resumed_ep_count
    stall_eps       = resumed_stall_eps
    start_update    = update
    start_step      = global_step
    start_time      = time.time()

    obs_buf  = torch.zeros(ROLLOUT_LEN, d_model).to(DEVICE)
    act_buf  = torch.zeros(ROLLOUT_LEN, 2).to(DEVICE)
    logp_buf = torch.zeros(ROLLOUT_LEN).to(DEVICE)
    rew_buf  = torch.zeros(ROLLOUT_LEN).to(DEVICE)
    val_buf  = torch.zeros(ROLLOUT_LEN).to(DEVICE)
    done_buf = torch.zeros(ROLLOUT_LEN).to(DEVICE)

    print('Starting rollout...')

    while global_step < total_steps:

        for t in range(ROLLOUT_LEN):
            frames_t, proprio_t = get_obs_tensor(env, raw_obs, frame_buf, DEVICE)
            with torch.no_grad():
                latent    = encoder(frames_t, proprio_t)
                dist, val = ppo(latent)
                action    = dist.sample()
                logp      = dist.log_prob(action).sum(-1)

            obs_buf[t]  = latent.squeeze(0)
            act_buf[t]  = action.squeeze(0)
            logp_buf[t] = logp.squeeze(0)
            val_buf[t]  = val.squeeze(0)

            action_np = np.clip(action.squeeze(0).cpu().numpy(), -1.0, 1.0)
            if env.agent.speed_km_h < MIN_SPEED_KMH and action_np[1] < MIN_THROTTLE:
                action_np[1] = MIN_THROTTLE       # env-side only; PPO still trains on the sampled action
                ep_floor_steps += 1
            raw_obs, _, term, trunc, info = env.step(action_np.tolist())
            speed_kmh          = env.agent.speed_km_h
            reward, completion = compute_reward(info, prev_completion, speed_kmh)
            stall_count = stall_count + 1 if speed_kmh < STALL_SPEED_KMH else 0
            if stall_count >= STALL_STEPS and not (term or trunc):
                trunc = True
                stall_eps += 1
                reward -= STALL_TERMINAL
            prev_completion    = completion
            rew_buf[t]         = reward
            done_buf[t]        = float(term or trunc)
            ep_reward         += reward
            ep_len            += 1
            global_step       += 1

            if term or trunc:
                ep_count += 1
                if ep_count % 10 == 0:
                    sps = (global_step - start_step) / (time.time() - start_time)
                    print('  step=' + str(global_step).rjust(8) +
                          ' | ep=' + str(ep_count).rjust(5) +
                          ' | reward=' + str(round(ep_reward, 2)).rjust(7) +
                          ' | len=' + str(ep_len).rjust(5) +
                          ' | completion=' + str(round(completion * 100, 1)) + '%' +
                          ' | stalled=' + str(stall_eps) +
                          ' | floor=' + str(round(ep_floor_steps / ep_len * 100, 1)) + '%' +
                          ' | sps=' + str(int(sps)), flush=True)
                    metrics.append({
                        'step': global_step, 'ep': ep_count,
                        'ep_reward': ep_reward, 'ep_len': ep_len,
                        'completion': completion,
                        'floor_steps': ep_floor_steps,
                    })
                ep_reward       = 0.0
                ep_len          = 0
                prev_completion = 0.0
                stall_count     = 0
                ep_floor_steps  = 0
                frame_buf       = []
                raw_obs, _ = env.reset(seed=random.randint(1000, 1399))
                reset_command_state(env)

        with torch.no_grad():
            frames_t, proprio_t = get_obs_tensor(env, raw_obs, frame_buf, DEVICE)
            next_val = ppo.get_value(encoder(frames_t, proprio_t)).squeeze(0)

        advantages = torch.zeros(ROLLOUT_LEN).to(DEVICE)
        last_gae   = 0.0
        for t in reversed(range(ROLLOUT_LEN)):
            nv       = next_val if t == ROLLOUT_LEN - 1 else val_buf[t + 1]
            delta    = rew_buf[t] + GAMMA * nv * (1 - done_buf[t]) - val_buf[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * (1 - done_buf[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + val_buf

        idx = np.arange(ROLLOUT_LEN)
        for _ in range(N_EPOCHS):
            np.random.shuffle(idx)
            for start in range(0, ROLLOUT_LEN, MINIBATCH):
                mb       = idx[start:start + MINIBATCH]
                mb_adv   = advantages[mb]
                mb_adv   = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                dist, val = ppo(obs_buf[mb])
                new_logp  = dist.log_prob(act_buf[mb]).sum(-1)
                entropy   = dist.entropy().sum(-1).mean()
                ratio     = (new_logp - logp_buf[mb]).exp()
                pg_loss   = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS)
                ).mean()
                vf_loss   = ((val - returns[mb]) ** 2).mean()
                loss      = pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ppo.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        update += 1

        if update % LOG_INTERVAL == 0:
            elapsed = (time.time() - start_time) / 60
            print('[update ' + str(update).rjust(5) + ']' +
                  ' step=' + str(global_step).rjust(8) +
                  ' | loss=' + str(round(loss.item(), 4)) +
                  ' | pg=' + str(round(pg_loss.item(), 4)) +
                  ' | vf=' + str(round(vf_loss.item(), 4)) +
                  ' | ent=' + str(round(entropy.item(), 4)) +
                  ' | elapsed=' + str(round(elapsed, 1)) + 'min', flush=True)

        if update % SAVE_INTERVAL == 0:
            torch.save({
                'update': update, 'step': global_step,
                'ppo': ppo.state_dict(),
                'optimizer': optimizer.state_dict(),
                'metrics': metrics,
                'ep_count': ep_count, 'stall_eps': stall_eps,
            }, out_dir / ('ppo_' + arm + '_update' + str(update) + '.pt'))
            with open(out_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f)
            print('  saved update ' + str(update), flush=True)
            # training slows down inside a long-lived process; exit so run_arm.sh restarts from this checkpoint
            if restart_every and update - start_update >= restart_every and global_step < total_steps:
                env.close()
                print('  exiting for scheduled restart after update ' + str(update), flush=True)
                sys.exit(RESTART_EXIT_CODE)

    env.close()
    torch.save({'ppo': ppo.state_dict(), 'metrics': metrics},
               out_dir / ('ppo_' + arm + '_final.pt'))
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f)
    print('Done — saved to ' + str(out_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', default='a0', choices=['a0', 'a1'])
    parser.add_argument('--total-steps', type=int, default=TOTAL_STEPS)
    parser.add_argument('--out-dir', default=str(OUT_DIR))
    parser.add_argument('--restart-every', type=int, default=0,
                        help='exit with code 3 after this many updates (at a checkpoint) so a wrapper can restart; 0 = never')
    args = parser.parse_args()
    train(args.arm, args.total_steps, args.out_dir, args.restart_every)
