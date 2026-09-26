"""
Deep representation analysis: A0 (pure BC) vs A1 (predictive BC) encoders.

Extends scripts/representation_analysis.py (linear probes, effective/stable
rank, whole-encoder CKA, RSA — not repeated here) with:

  1. Layer-wise CKA           — where in the CNN do A0 / A1 diverge?
  2. t-SNE                    — does A1 organise its feature space by
                                 navigation intent (command) better than A0?
  3. Dead neuron analysis     — how many of the 256 output dims are inert?
  4. Geometry preservation    — does feature-space distance track
                                 task-space (lat_offset) distance?
  5. IDM-expert linear probes — same probes as the base script, but on
                                 on-distribution (expert-driven) states
                                 instead of random-action rollouts.

Usage:
    python scripts/deep_representation_analysis.py [--force]

--force re-collects raw_frames.npy / *_idm.npy even if cached copies exist.
"""
import sys, os, argparse, json
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.manifold import TSNE
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)
torch.manual_seed(0)

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from src.models.policy import VisualPolicy
from src.env.metadrive_wrapper import (
    TEST_CFG, build_student_obs, reset_command_state, get_privileged_labels,
)

REP_DIR = "representation"
CKPT_DIR = "checkpoints"
DEVICE = torch.device("cpu")

C_A0 = "#e74c3c"
C_A1 = "#2ecc71"
BG = "#0f0f0f"
AX_BG = "#1a1a1a"


# ── style helper (matches representation_analysis.py) ──────────────────────
def style_axis(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors="#cccccc")
    ax.xaxis.label.set_color("#cccccc")
    ax.yaxis.label.set_color("#cccccc")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")


# ── encoder loading ──────────────────────────────────────────────────────
def load_policy(arm: str) -> VisualPolicy:
    ckpt = torch.load(f"{CKPT_DIR}/{arm}_500k_s0_final.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    policy = VisualPolicy(
        chunk_len=cfg.get("chunk_len", 8), K=cfg.get("K", 8),
        d_model=cfg.get("d_model", 256), n_layers=cfg.get("n_layers", 4),
        n_heads=cfg.get("n_heads", 4),
    )
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy


class FrozenEncoder(nn.Module):
    """Same convention as scripts/collect_features.py: mean-pool the K
    per-frame CNN latents into a single (B, 256) vector."""
    def __init__(self, policy: VisualPolicy):
        super().__init__()
        self.encoder = policy.encoder
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, frames):
        B, K, C, H, W = frames.shape
        x = frames.reshape(B * K, C, H, W).float() / 255.0
        x = self.encoder(x)
        return x.reshape(B, K, -1).mean(dim=1)


# ── MetaDrive rollout helpers ────────────────────────────────────────────
def make_eval_env(seed: int, agent_policy=None):
    cfg = {**TEST_CFG, "num_scenarios": 1, "start_seed": seed,
           "use_render": False, "preload_models": False}
    if agent_policy is not None:
        cfg["agent_policy"] = agent_policy
    return MetaDriveEnv(cfg)


def collect_raw_frame_stacks(seeds, n_frames, K=8, max_steps_per_seed=60):
    """Collect n_frames independent K-length history stacks via a random
    (non-expert) rollout — used only to probe raw conv-layer activations,
    so driving quality doesn't matter here."""
    stacks = []
    seed_i = 0
    while len(stacks) < n_frames:
        seed = seeds[seed_i % len(seeds)]
        seed_i += 1
        env = make_eval_env(seed)
        raw_obs, _ = env.reset()
        reset_command_state(env)
        frame_buf = []
        for _ in range(max_steps_per_seed):
            student = build_student_obs(env, raw_obs)
            frame = torch.from_numpy(student.image.transpose(2, 0, 1).copy())
            frame_buf.append(frame)
            if len(frame_buf) > K:
                frame_buf.pop(0)
            while len(frame_buf) < K:
                frame_buf.insert(0, frame_buf[0])

            stacks.append(torch.stack(frame_buf, dim=0).numpy())  # (K,3,84,84)
            if len(stacks) >= n_frames:
                break

            action = [np.random.uniform(-0.5, 0.5), np.random.uniform(0.2, 0.6)]
            raw_obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        env.close()
    return np.stack(stacks[:n_frames], axis=0)  # (n_frames, K, 3, 84, 84) uint8


def collect_idm_rollout(n_frames, K=8, start_seed=3000, max_steps_per_episode=400):
    """One shared IDM-expert-driven rollout — states are on-distribution
    (proper lane following). Collected ONCE and reused for both encoders
    so A0 vs A1 probes are computed on identical inputs."""
    frames_out, lat_off, heading_err, speed, command = [], [], [], [], []
    seed = start_seed

    while len(frames_out) < n_frames:
        env = make_eval_env(seed, agent_policy=IDMPolicy)
        raw_obs, _ = env.reset()
        reset_command_state(env)
        frame_buf = []

        for _ in range(max_steps_per_episode):
            student = build_student_obs(env, raw_obs)
            frame = torch.from_numpy(student.image.transpose(2, 0, 1).copy())
            frame_buf.append(frame)
            if len(frame_buf) > K:
                frame_buf.pop(0)
            while len(frame_buf) < K:
                frame_buf.insert(0, frame_buf[0])

            frames_out.append(torch.stack(frame_buf, dim=0).numpy())
            try:
                priv = get_privileged_labels(env)
                lat_off.append(priv["lat_offset"])
                heading_err.append(priv["heading_err"])
                speed.append(priv["speed"])
            except Exception:
                lat_off.append(0.0); heading_err.append(0.0); speed.append(0.0)
            command.append(student.command)

            if len(frames_out) >= n_frames:
                break

            # agent_policy=IDMPolicy drives the car; the passed action is a
            # required placeholder and is ignored (same convention as
            # scripts/test_expert.py).
            raw_obs, _, term, trunc, info = env.step([0.0, 0.0])
            if term or trunc:
                break

        env.close()
        seed += 1
        print(f"  seed={seed - 1} -> {len(frames_out)}/{n_frames} IDM frames collected")

    frames_arr = np.stack(frames_out[:n_frames], axis=0)
    labels = {
        "lat_offset": np.array(lat_off[:n_frames]),
        "heading_err": np.array(heading_err[:n_frames]),
        "speed": np.array(speed[:n_frames]),
        "command": np.array(command[:n_frames]),
    }
    return frames_arr, labels


# ── 1. layer-wise CKA ────────────────────────────────────────────────────
class _Recorder:
    def __init__(self):
        self.out = None
    def __call__(self, module, inp, output):
        self.out = output.detach()


def get_conv_layer_activations(encoder: nn.Module, x: torch.Tensor, batch_size=128):
    """Register forward hooks on every Conv2d in encoder.conv and return
    the concatenated activations for each layer, in order."""
    conv_layers = [m for m in encoder.conv if isinstance(m, nn.Conv2d)]
    recorders = [_Recorder() for _ in conv_layers]
    handles = [layer.register_forward_hook(rec) for layer, rec in zip(conv_layers, recorders)]

    per_layer_batches = [[] for _ in conv_layers]
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            batch = x[i:i + batch_size]
            encoder.conv(batch)
            for li, rec in enumerate(recorders):
                per_layer_batches[li].append(rec.out.numpy())

    for h in handles:
        h.remove()
    return [np.concatenate(b, axis=0) for b in per_layer_batches]


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA via centered N×N Gram matrices (Kornblith et al. 2019).
    Memory cost is O(N^2), independent of feature dimensionality — required
    here since early conv layers have tens of thousands of channels*H*W."""
    X = X.reshape(X.shape[0], -1).astype(np.float64)
    Y = Y.reshape(Y.shape[0], -1).astype(np.float64)

    def center(K):
        n = K.shape[0]
        unit = np.ones((n, n)) / n
        return K - unit @ K - K @ unit + unit @ K @ unit

    Kx = center(X @ X.T)
    Ky = center(Y @ Y.T)
    hsic = (Kx * Ky).sum()
    denom = np.sqrt((Kx * Kx).sum()) * np.sqrt((Ky * Ky).sum())
    return float(hsic / denom) if denom > 0 else 0.0


def run_layer_wise_cka(policy_a0, policy_a1, raw_frames: np.ndarray):
    n, K, C, H, W = raw_frames.shape
    x = torch.from_numpy(raw_frames.reshape(n * K, C, H, W)).float() / 255.0

    print("  extracting A0 conv-layer activations...")
    acts_a0 = get_conv_layer_activations(policy_a0.encoder, x)
    print("  extracting A1 conv-layer activations...")
    acts_a1 = get_conv_layer_activations(policy_a1.encoder, x)

    n_layers = len(acts_a0)
    ckas = []
    for li in range(n_layers):
        cka = linear_cka(acts_a0[li], acts_a1[li])
        ckas.append(cka)
        print(f"    layer {li + 1}/{n_layers}  shape={acts_a0[li].shape[1:]}  CKA={cka:.4f}")
    return ckas


# ── 2. t-SNE ─────────────────────────────────────────────────────────────
def run_tsne(X: np.ndarray, n_samples=1000):
    n = min(n_samples, X.shape[0])
    idx = np.random.choice(X.shape[0], n, replace=False)
    emb = TSNE(n_components=2, init="pca", learning_rate="auto",
               perplexity=min(30, n - 1), random_state=0).fit_transform(X[idx])
    return emb, idx


# ── 3. dead neuron analysis ──────────────────────────────────────────────
def dead_neuron_analysis(X: np.ndarray, threshold=0.01):
    """
    Per-dimension variance, normalised by the encoder's own mean per-dim
    variance (NOT sklearn's StandardScaler — that forces every dimension's
    post-transform variance to exactly 1.0, which would make any variance
    threshold vacuous). A dimension is "dead" if its variance is under 1%
    of this encoder's typical dimension's variance.
    """
    var = X.var(axis=0)
    norm_var = var / (var.mean() + 1e-12)
    dead_mask = norm_var < threshold
    return var, norm_var, dead_mask


# ── 4. geometry preservation ─────────────────────────────────────────────
def geometry_preservation_score(X: np.ndarray, task_values: np.ndarray, n_pairs=500):
    n = X.shape[0]
    i = np.random.randint(0, n, n_pairs)
    j = np.random.randint(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]

    feat_dist = np.linalg.norm(X[i] - X[j], axis=1)
    task_dist = np.abs(task_values[i] - task_values[j])
    rho, p = spearmanr(feat_dist, task_dist)
    return float(rho), float(p)


# ── 5. IDM-expert linear probes ──────────────────────────────────────────
def run_linear_probes(X_a0, y_a0, X_a1, y_a1):
    """
    NOTE on CV splitting: unlike the base script's random-action features
    (which reset episodes constantly and so were fairly well mixed), the
    IDM rollout is one continuous sequence of long, smooth episodes
    concatenated back-to-back — lat_offset/heading_err variance differs
    drastically between contiguous chunks (different seeds/episodes drift
    differently). Plain `cv=5` uses non-shuffled KFold, so folds end up
    training on one episode's error regime and testing on another's,
    which made Ridge extrapolate wildly (R2 as low as -60). Explicit
    shuffled splitters fix this.
    """
    results = {}
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    for name in ("lat_offset", "heading_err", "speed"):
        r2_a0 = cross_val_score(Ridge(alpha=1.0), X_a0, y_a0[name], cv=kfold, scoring="r2").mean()
        r2_a1 = cross_val_score(Ridge(alpha=1.0), X_a1, y_a1[name], cv=kfold, scoring="r2").mean()
        results[name] = {"a0": float(r2_a0), "a1": float(r2_a1)}
        print(f"  {name:<15} A0: R2={r2_a0:.3f}   A1: R2={r2_a1:.3f}   delta={r2_a1 - r2_a0:+.3f}")

    cmd_a0 = LabelEncoder().fit_transform(y_a0["command"])
    cmd_a1 = LabelEncoder().fit_transform(y_a1["command"])
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    acc_a0 = cross_val_score(LogisticRegression(max_iter=1000), X_a0, cmd_a0, cv=skfold, scoring="accuracy").mean()
    acc_a1 = cross_val_score(LogisticRegression(max_iter=1000), X_a1, cmd_a1, cv=skfold, scoring="accuracy").mean()
    results["command"] = {"a0": float(acc_a0), "a1": float(acc_a1)}
    print(f"  {'command':<15} A0: Acc={acc_a0:.3f}   A1: Acc={acc_a1:.3f}   delta={acc_a1 - acc_a0:+.3f}")
    return results


# ── main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                         help="re-collect raw_frames.npy / *_idm.npy even if cached")
    args = parser.parse_args()

    os.makedirs(REP_DIR, exist_ok=True)

    print("Loading existing features + labels...")
    a0_feat = np.load(f"{REP_DIR}/a0_features.npy")
    a1_feat = np.load(f"{REP_DIR}/a1_features.npy")
    a0_labels = np.load(f"{REP_DIR}/a0_labels.npy", allow_pickle=True).item()
    a1_labels = np.load(f"{REP_DIR}/a1_labels.npy", allow_pickle=True).item()
    print(f"  A0: {a0_feat.shape}   A1: {a1_feat.shape}")

    scaler_a0 = StandardScaler().fit(a0_feat)
    scaler_a1 = StandardScaler().fit(a1_feat)
    X_a0 = scaler_a0.transform(a0_feat)
    X_a1 = scaler_a1.transform(a1_feat)

    print("\nLoading A0 / A1 policies (for the raw CNN encoder)...")
    policy_a0 = load_policy("a0")
    policy_a1 = load_policy("a1")

    # ── 1. layer-wise CKA ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. LAYER-WISE CKA")
    print("=" * 60)

    raw_frames_path = f"{REP_DIR}/raw_frames.npy"
    if os.path.exists(raw_frames_path) and not args.force:
        print(f"  loading cached {raw_frames_path}")
        raw_frames = np.load(raw_frames_path)
    else:
        print("  collecting 200 raw frame stacks from seeds 3000-3005...")
        raw_frames = collect_raw_frame_stacks(seeds=list(range(3000, 3006)), n_frames=200)
        np.save(raw_frames_path, raw_frames)
        print(f"  saved {raw_frames_path}  shape={raw_frames.shape}")

    layer_ckas = run_layer_wise_cka(policy_a0, policy_a1, raw_frames)

    # ── 2. t-SNE ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. t-SNE")
    print("=" * 60)
    tsne_a0, idx_a0 = run_tsne(X_a0)
    tsne_a1, idx_a1 = run_tsne(X_a1)
    print(f"  A0 t-SNE on {len(idx_a0)} samples, A1 on {len(idx_a1)} samples")

    # ── 3. dead neuron analysis ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. DEAD NEURON ANALYSIS")
    print("=" * 60)
    var_a0, norm_var_a0, dead_a0 = dead_neuron_analysis(a0_feat)
    var_a1, norm_var_a1, dead_a1 = dead_neuron_analysis(a1_feat)
    print(f"  A0: {dead_a0.sum()} / {len(dead_a0)} dead dimensions (< 1% of mean variance)")
    print(f"  A1: {dead_a1.sum()} / {len(dead_a1)} dead dimensions (< 1% of mean variance)")

    # ── 4. geometry preservation ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. GEOMETRY PRESERVATION (feature distance vs |lat_offset| distance)")
    print("=" * 60)
    rho_a0, p_a0 = geometry_preservation_score(X_a0, a0_labels["lat_offset"])
    rho_a1, p_a1 = geometry_preservation_score(X_a1, a1_labels["lat_offset"])
    print(f"  A0: Spearman rho={rho_a0:.4f}  (p={p_a0:.2e})")
    print(f"  A1: Spearman rho={rho_a1:.4f}  (p={p_a1:.2e})")
    print(f"  {'A1 better' if rho_a1 > rho_a0 else 'A0 better'} by {abs(rho_a1 - rho_a0):.4f}")

    # ── 5. IDM-expert linear probes ────────────────────────────────────
    print("\n" + "=" * 60)
    print("5. IDM-EXPERT LINEAR PROBES (on-distribution states)")
    print("=" * 60)

    idm_a0_feat_path = f"{REP_DIR}/a0_features_idm.npy"
    idm_a1_feat_path = f"{REP_DIR}/a1_features_idm.npy"
    idm_a0_lab_path = f"{REP_DIR}/a0_labels_idm.npy"
    idm_a1_lab_path = f"{REP_DIR}/a1_labels_idm.npy"

    if all(os.path.exists(p) for p in
           (idm_a0_feat_path, idm_a1_feat_path, idm_a0_lab_path, idm_a1_lab_path)) and not args.force:
        print("  loading cached IDM features/labels")
        a0_feat_idm = np.load(idm_a0_feat_path)
        a1_feat_idm = np.load(idm_a1_feat_path)
        a0_labels_idm = np.load(idm_a0_lab_path, allow_pickle=True).item()
        a1_labels_idm = np.load(idm_a1_lab_path, allow_pickle=True).item()
    else:
        print("  collecting 3000 frames from a single shared IDM-expert rollout...")
        idm_frames, idm_labels = collect_idm_rollout(n_frames=3000, start_seed=3000)

        enc_a0 = FrozenEncoder(policy_a0).to(DEVICE)
        enc_a1 = FrozenEncoder(policy_a1).to(DEVICE)

        feats_a0, feats_a1 = [], []
        batch_size = 128
        x_all = torch.from_numpy(idm_frames)
        with torch.no_grad():
            for i in range(0, x_all.shape[0], batch_size):
                batch = x_all[i:i + batch_size]
                feats_a0.append(enc_a0(batch).numpy())
                feats_a1.append(enc_a1(batch).numpy())
        a0_feat_idm = np.concatenate(feats_a0, axis=0)
        a1_feat_idm = np.concatenate(feats_a1, axis=0)
        a0_labels_idm = idm_labels
        a1_labels_idm = idm_labels

        np.save(idm_a0_feat_path, a0_feat_idm)
        np.save(idm_a1_feat_path, a1_feat_idm)
        np.save(idm_a0_lab_path, a0_labels_idm)
        np.save(idm_a1_lab_path, a1_labels_idm)
        print(f"  saved IDM features: A0={a0_feat_idm.shape}  A1={a1_feat_idm.shape}")

    X_a0_idm = StandardScaler().fit_transform(a0_feat_idm)
    X_a1_idm = StandardScaler().fit_transform(a1_feat_idm)
    idm_probe_results = run_linear_probes(X_a0_idm, a0_labels_idm, X_a1_idm, a1_labels_idm)

    with open(f"{REP_DIR}/idm_probe_results.json", "w") as f:
        json.dump(idm_probe_results, f, indent=2)
    print(f"  saved {REP_DIR}/idm_probe_results.json")

    # ── combined figure ──────────────────────────────────────────────────
    print("\nBuilding combined figure...")
    fig = plt.figure(figsize=(22, 15))
    fig.patch.set_facecolor(BG)
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.35)

    # -- layer-wise CKA --
    ax = fig.add_subplot(gs[0, 0:2])
    style_axis(ax)
    layers_x = np.arange(1, len(layer_ckas) + 1)
    ax.plot(layers_x, layer_ckas, marker="o", color="#3498db", linewidth=2, markersize=7)
    ax.set_xlabel("Conv layer depth")
    ax.set_ylabel("Linear CKA (A0 vs A1)")
    ax.set_title("Layer-wise CKA: where do A0 / A1 diverge?", fontweight="bold")
    ax.set_xticks(layers_x)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2, color="#444444")

    # -- geometry preservation bar chart --
    ax = fig.add_subplot(gs[0, 2:4])
    style_axis(ax)
    bars = ax.bar(["A0\n(Pure BC)", "A1\n(Predictive)"], [rho_a0, rho_a1],
                   color=[C_A0, C_A1], width=0.5)
    ax.set_ylabel("Spearman rho (feature dist vs |lat_offset| dist)")
    ax.set_title("Geometry Preservation", fontweight="bold")
    ax.grid(alpha=0.2, color="#444444", axis="y")
    for bar, val in zip(bars, [rho_a0, rho_a1]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", color="white", fontweight="bold")

    # -- t-SNE, 4 panels --
    cmd_cmap = matplotlib.colors.ListedColormap(["#3498db", "#e67e22", "#9b59b6"])
    cmd_names = ["straight", "right", "left"]

    ax = fig.add_subplot(gs[1, 0]); style_axis(ax)
    cmd_a0 = a0_labels["command"][idx_a0]
    sc = ax.scatter(tsne_a0[:, 0], tsne_a0[:, 1], c=cmd_a0, cmap=cmd_cmap, s=8, vmin=0, vmax=2)
    ax.set_title("A0 — t-SNE by command", fontweight="bold", fontsize=11)

    ax = fig.add_subplot(gs[1, 1]); style_axis(ax)
    cmd_a1 = a1_labels["command"][idx_a1]
    ax.scatter(tsne_a1[:, 0], tsne_a1[:, 1], c=cmd_a1, cmap=cmd_cmap, s=8, vmin=0, vmax=2)
    ax.set_title("A1 — t-SNE by command", fontweight="bold", fontsize=11)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=cmd_cmap(i / 2), label=n)
               for i, n in enumerate(cmd_names)]
    ax.legend(handles=handles, facecolor="#2a2a2a", labelcolor="white", fontsize=8, loc="best")

    ax = fig.add_subplot(gs[1, 2]); style_axis(ax)
    spd_a0 = a0_labels["speed"][idx_a0]
    sc2 = ax.scatter(tsne_a0[:, 0], tsne_a0[:, 1], c=spd_a0, cmap="viridis", s=8)
    ax.set_title("A0 — t-SNE by speed", fontweight="bold", fontsize=11)
    plt.colorbar(sc2, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 3]); style_axis(ax)
    spd_a1 = a1_labels["speed"][idx_a1]
    sc3 = ax.scatter(tsne_a1[:, 0], tsne_a1[:, 1], c=spd_a1, cmap="viridis", s=8)
    ax.set_title("A1 — t-SNE by speed", fontweight="bold", fontsize=11)
    plt.colorbar(sc3, ax=ax, fraction=0.046, pad=0.04)

    # -- dead neuron histogram --
    ax = fig.add_subplot(gs[2, :])
    style_axis(ax)
    bins = np.logspace(np.log10(max(norm_var_a0.min(), 1e-6)),
                        np.log10(max(norm_var_a0.max(), norm_var_a1.max(), 1.0)), 40)
    ax.hist(norm_var_a0, bins=bins, color=C_A0, alpha=0.6, label=f"A0 ({dead_a0.sum()} dead)")
    ax.hist(norm_var_a1, bins=bins, color=C_A1, alpha=0.6, label=f"A1 ({dead_a1.sum()} dead)")
    ax.axvline(0.01, color="white", linestyle="--", linewidth=1, alpha=0.7, label="dead threshold (0.01)")
    ax.set_xscale("log")
    ax.set_xlabel("Per-dimension variance, normalised by mean variance")
    ax.set_ylabel("Number of dimensions")
    ax.set_title("Dead Neuron Analysis (256-dim encoder output)", fontweight="bold")
    ax.legend(facecolor="#2a2a2a", labelcolor="white", fontsize=9)
    ax.grid(alpha=0.2, color="#444444")

    plt.suptitle("Deep Representation Analysis: A0 vs A1 Encoder",
                 fontsize=17, fontweight="bold", color="white", y=0.995)
    out_path = f"{REP_DIR}/deep_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved combined figure to {out_path}")

    # ── final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Layer-wise CKA (A0 vs A1): {[f'{c:.3f}' for c in layer_ckas]}")
    print(f"  Dead dims — A0: {dead_a0.sum()}/256   A1: {dead_a1.sum()}/256")
    print(f"  Geometry preservation (Spearman rho) — A0: {rho_a0:.4f}   A1: {rho_a1:.4f}")
    print("  IDM-expert probes:")
    for k, v in idm_probe_results.items():
        diff = v["a1"] - v["a0"]
        print(f"    {k:<15}: A1 {'better' if diff > 0 else 'worse'} by {abs(diff):.3f}")


if __name__ == "__main__":
    main()
