import sys, os
sys.path.insert(0, '.')

import torch
import numpy as np
from pathlib import Path
from metadrive.envs.metadrive_env import MetaDriveEnv
from src.models.policy import VisualPolicy
from src.env.metadrive_wrapper import TEST_CFG, build_student_obs, reset_command_state, get_privileged_labels

K = 8
N_EPISODES = 30
MAX_STEPS  = 300
SEEDS      = list(range(3000, 3030))
class FrozenEncoder(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.encoder = policy.encoder
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, frames):
        B, Kf, C, H, W = frames.shape
        x = frames.reshape(B * Kf, C, H, W).float() / 255.0
        x = self.encoder(x)
        x = x.reshape(B, Kf, -1).mean(dim=1)
        return x

device = torch.device('cpu')

eval_cfg = {
    **TEST_CFG,
    'num_scenarios':  1,
    'use_render':     False,
    'preload_models': False,
}

Path('representation').mkdir(exist_ok=True)

for arm in ['a0', 'a1']:
    print(f"\nCollecting features for {arm}...")

    ckpt   = torch.load(f'checkpoints/{arm}_500k_s0_final.pt',
                        map_location='cpu', weights_only=False)
    cfg    = ckpt['config']
    policy = VisualPolicy(
        chunk_len=cfg.get('chunk_len', 8), K=cfg.get('K', 8),
        d_model=cfg.get('d_model', 256),   n_layers=cfg.get('n_layers', 4),
        n_heads=cfg.get('n_heads', 4),
    )
    policy.load_state_dict(ckpt['policy'])
    encoder = FrozenEncoder(policy).to(device)
    encoder.eval()

    all_features  = []
    all_lat_off   = []
    all_heading   = []
    all_speed     = []
    all_command   = []

    for seed in SEEDS:
        seed_cfg = {**eval_cfg, 'start_seed': seed}
        env      = MetaDriveEnv(seed_cfg)
        raw_obs, _ = env.reset()
        reset_command_state(env)
        frame_buf = []

        for step in range(MAX_STEPS):
            student = build_student_obs(env, raw_obs)
            frame   = torch.from_numpy(student.image.transpose(2, 0, 1).copy()).unsqueeze(0)
            frame_buf.append(frame)
            if len(frame_buf) > K:    frame_buf.pop(0)
            while len(frame_buf) < K: frame_buf.insert(0, frame_buf[0])

            frames_t = torch.cat(frame_buf, dim=0).unsqueeze(0)
            with torch.no_grad():
                feat = encoder(frames_t)  # (1, 256)

            # get privileged labels
            try:
                priv = get_privileged_labels(env)
                lat  = priv['lat_offset']
                head = priv['heading_err']
                spd  = priv['speed']
            except:
                lat, head, spd = 0.0, 0.0, 0.0

            all_features.append(feat.squeeze(0).numpy())
            all_lat_off.append(lat)
            all_heading.append(head)
            all_speed.append(spd)
            all_command.append(student.command)

            action = [np.random.uniform(-0.5, 0.5), np.random.uniform(0.2, 0.6)]
            raw_obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break

        env.close()
        print(f"  seed={seed} — {len(all_features)} features collected so far")

    features = np.array(all_features)   # (N, 256)
    labels   = {
        'lat_offset':  np.array(all_lat_off),
        'heading_err': np.array(all_heading),
        'speed':       np.array(all_speed),
        'command':     np.array(all_command),
    }

    np.save(f'representation/{arm}_features.npy', features)
    np.save(f'representation/{arm}_labels.npy',   labels)
    print(f"Saved {features.shape[0]} feature vectors for {arm}")

print("\nDone — features saved to representation/")