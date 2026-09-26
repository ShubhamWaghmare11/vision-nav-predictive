import sys, os, json, torch, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.policy import VisualPolicy
from src.env.metadrive_wrapper import TEST_CFG
from pathlib import Path

runs = [
    ('a0', '50k',  0), ('a0', '50k',  1), ('a0', '50k',  2),
    ('a1', '50k',  0), ('a1', '50k',  1), ('a1', '50k',  2),
    ('a0', '150k', 0), ('a0', '150k', 1), ('a0', '150k', 2),
    ('a1', '150k', 0), ('a1', '150k', 1), ('a1', '150k', 2),
    ('a0', '500k', 0), ('a0', '500k', 1), ('a0', '500k', 2),
    ('a1', '500k', 0), ('a1', '500k', 1), ('a1', '500k', 2),
    ('a2', '500k', 0), ('a2', '500k', 1), ('a2', '500k', 2),
    ('a3', '500k', 0), ('a3', '500k', 1), ('a3', '500k', 2),
]

eval_cfg = {
    **TEST_CFG,
    'num_scenarios':  100,
    'use_render':     True,
    'preload_models': False,
}

device      = torch.device('cpu')
test_seeds  = list(range(3000, 3050))
ckpt_dir    = Path('checkpoints')
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

total_runs = len(runs)

for run_idx, (arm, regime, seed) in enumerate(runs):
    run_name  = f"{arm}_{regime}_s{seed}"
    ckpt_path = ckpt_dir / f"{run_name}_final.pt"
    out_path  = results_dir / f"{run_name}.json"

    print(f"\n[{run_idx+1}/{total_runs}] {run_name}")

    if out_path.exists():
        print(f"  SKIP (already done)")
        continue

    if not ckpt_path.exists():
        print(f"  MISSING checkpoint: {ckpt_path}")
        continue

    print(f"  Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg  = ckpt['config']

    policy = VisualPolicy(
        chunk_len = cfg.get('chunk_len', 8),
        K         = cfg.get('K', 8),
        d_model   = cfg.get('d_model', 256),
        n_layers  = cfg.get('n_layers', 4),
        n_heads   = cfg.get('n_heads', 4),
    ).to(device)
    policy.load_state_dict(ckpt['policy'])
    print(f"  Checkpoint loaded")

    from metadrive.envs.metadrive_env import MetaDriveEnv
    from src.env.metadrive_wrapper import build_student_obs, reset_command_state, get_privileged_labels
    import numpy as np

    print(f"  Creating env...")
    env = MetaDriveEnv(eval_cfg)
    print(f"  Env created, running {len(test_seeds)} episodes...")

    all_completion, all_success, all_xte, all_jerk, all_km, all_infraction = [], [], [], [], [], []
    run_start = time.time()

    for ep_idx, ep_seed in enumerate(test_seeds):
        ep_start = time.time()

        raw_obs, _ = env.reset(seed=ep_seed)
        reset_command_state(env)

        frame_buf     = []
        steers        = []
        xtes          = []
        infractions   = 0
        km_driven     = 0.0
        completion    = 0.0
        success       = False
        current_chunk = None
        chunk_pos     = 0
        K             = cfg.get('K', 8)
        exec_len      = 4
        term_reason   = 'unknown'

        for step in range(1000):
            student = build_student_obs(env, raw_obs)

            frame_buf.append(
                torch.from_numpy(student.image.transpose(2, 0, 1).copy()).unsqueeze(0)
            )
            if len(frame_buf) > K:
                frame_buf.pop(0)
            while len(frame_buf) < K:
                frame_buf.insert(0, frame_buf[0])

            if current_chunk is None or chunk_pos >= exec_len:
                frames_t  = torch.cat(frame_buf, dim=0).unsqueeze(0).to(device)
                proprio_t = torch.tensor(student.proprio, dtype=torch.float32).unsqueeze(0).to(device)
                command_t = torch.tensor([student.command], dtype=torch.long).to(device)
                with torch.no_grad():
                    actions, _ = policy(frames_t, proprio_t, command_t)
                current_chunk = actions[0].cpu().numpy()
                chunk_pos = 0

            action    = current_chunk[chunk_pos]
            chunk_pos += 1
            steers.append(float(action[0]))

            raw_obs, _, term, trunc, info = env.step(action.tolist())

            try:
                priv = get_privileged_labels(env)
                xtes.append(abs(priv["lat_offset"]))
            except Exception:
                xtes.append(0.0)

            speed      = float(env.agent.speed)
            km_driven += speed * 0.02 * 5 / 1000.0
            completion = float(info.get("route_completion", 0.0))

            if info.get("crash_vehicle", False) or info.get("crash_object", False) or info.get("out_of_road", False):
                infractions += 1
            if info.get("arrive_dest", False):
                success = True

            if term or trunc:
                reasons = []
                if info.get('out_of_road'):   reasons.append('out_of_road')
                if info.get('crash_vehicle'): reasons.append('crash_vehicle')
                if info.get('crash_object'):  reasons.append('crash_object')
                if info.get('arrive_dest'):   reasons.append('arrived')
                if trunc:                     reasons.append('timeout')
                term_reason = ','.join(reasons) if reasons else 'unknown'
                break

        if len(steers) >= 3:
            jerk_vals = [abs(steers[i] - 2*steers[i-1] + steers[i-2]) for i in range(2, len(steers))]
            jerk = float(np.mean(jerk_vals))
        else:
            jerk = 0.0

        all_completion.append(completion)
        all_success.append(float(success))
        all_xte.append(float(np.mean(xtes)) if xtes else 0.0)
        all_jerk.append(jerk)
        all_km.append(km_driven)
        all_infraction.append(infractions)

        ep_time = time.time() - ep_start
        elapsed = time.time() - run_start
        eta     = (elapsed / (ep_idx + 1)) * (len(test_seeds) - ep_idx - 1)

        print(f"  ep {ep_idx+1:02d}/{len(test_seeds)} | seed={ep_seed} | "
              f"done={completion:.0%} | {'✅' if success else '❌'} | "
              f"xte={all_xte[-1]:.2f} | {ep_time:.0f}s | {term_reason} | eta={eta/60:.1f}min",
              flush=True)

    env.close()

    total_km          = sum(all_km)
    total_infractions = sum(r * k for r, k in zip(all_infraction, all_km))

    results = {
        'run_name':           run_name,
        'arm':                arm,
        'regime':             regime,
        'seed':               seed,
        'completion_mean':    float(np.mean(all_completion)),
        'completion_std':     float(np.std(all_completion)),
        'success_rate':       float(np.mean(all_success)),
        'infractions_per_km': float(total_infractions / max(total_km, 0.001)),
        'mean_xte':           float(np.mean(all_xte)),
        'steering_jerk':      float(np.mean(all_jerk)),
        'n_episodes':         len(test_seeds),
        'per_episode': {
            'completion': all_completion,
            'success':    all_success,
            'xte':        all_xte,
            'jerk':       all_jerk,
        }
    }

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  ── {run_name} done ──")
    print(f"  completion: {results['completion_mean']:.1%} ± {results['completion_std']:.1%}")
    print(f"  success:    {results['success_rate']:.1%}")
    print(f"  xte:        {results['mean_xte']:.3f}")
    print(f"  saved:      {out_path}")

print("\n" + "="*50)
print("ALL EVALUATIONS DONE")
print("="*50)