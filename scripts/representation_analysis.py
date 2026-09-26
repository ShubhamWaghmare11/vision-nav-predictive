import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

# ── load features ──────────────────────────────────────────────────────────────
a0_feat   = np.load('representation/a0_features.npy')
a1_feat   = np.load('representation/a1_features.npy')
a0_labels = np.load('representation/a0_labels.npy', allow_pickle=True).item()
a1_labels = np.load('representation/a1_labels.npy', allow_pickle=True).item()

print(f"A0 features: {a0_feat.shape}")
print(f"A1 features: {a1_feat.shape}")
print()

scaler_a0 = StandardScaler().fit(a0_feat)
scaler_a1 = StandardScaler().fit(a1_feat)
X_a0 = scaler_a0.transform(a0_feat)
X_a1 = scaler_a1.transform(a1_feat)

# ── 1. Linear Probes ───────────────────────────────────────────────────────────
print("=" * 50)
print("1. LINEAR PROBES (R² via 5-fold CV)")
print("=" * 50)

probe_results = {}

for label_name in ['lat_offset', 'heading_err', 'speed']:
    y_a0 = a0_labels[label_name]
    y_a1 = a1_labels[label_name]
    r2_a0 = cross_val_score(Ridge(alpha=1.0), X_a0, y_a0, cv=5, scoring='r2').mean()
    r2_a1 = cross_val_score(Ridge(alpha=1.0), X_a1, y_a1, cv=5, scoring='r2').mean()
    probe_results[label_name] = {'a0': r2_a0, 'a1': r2_a1}
    print(f"  {label_name:<15} A0: R²={r2_a0:.3f}   A1: R²={r2_a1:.3f}   Δ={r2_a1-r2_a0:+.3f}")

y_cmd_a0 = LabelEncoder().fit_transform(a0_labels['command'])
y_cmd_a1 = LabelEncoder().fit_transform(a1_labels['command'])
acc_a0 = cross_val_score(LogisticRegression(max_iter=1000), X_a0, y_cmd_a0, cv=5, scoring='accuracy').mean()
acc_a1 = cross_val_score(LogisticRegression(max_iter=1000), X_a1, y_cmd_a1, cv=5, scoring='accuracy').mean()
probe_results['command'] = {'a0': acc_a0, 'a1': acc_a1}
print(f"  {'command':<15} A0: Acc={acc_a0:.3f}   A1: Acc={acc_a1:.3f}   Δ={acc_a1-acc_a0:+.3f}")

# ── 2. Effective Rank ──────────────────────────────────────────────────────────
print()
print("=" * 50)
print("2. EFFECTIVE RANK (Singular Value Analysis)")
print("=" * 50)

def effective_rank(X):
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    S = S / S.sum()
    entropy = -(S * np.log(S + 1e-10)).sum()
    return float(np.exp(entropy))

def stable_rank(X):
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    return float((S**2).sum() / (S**2).max())

er_a0 = effective_rank(X_a0)
er_a1 = effective_rank(X_a1)
sr_a0 = stable_rank(X_a0)
sr_a1 = stable_rank(X_a1)

print(f"  Effective rank:  A0={er_a0:.1f}   A1={er_a1:.1f}   Δ={er_a1-er_a0:+.1f}")
print(f"  Stable rank:     A0={sr_a0:.1f}   A1={sr_a1:.1f}   Δ={sr_a1-sr_a0:+.1f}")

_, S_a0, _ = np.linalg.svd(X_a0, full_matrices=False)
_, S_a1, _ = np.linalg.svd(X_a1, full_matrices=False)

# ── 3. CKA ────────────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("3. CKA (Representational Similarity)")
print("=" * 50)

def linear_cka(X, Y):
    # use min size for fair comparison
    n = min(len(X), len(Y))
    X = X[:n] - X[:n].mean(0)
    Y = Y[:n] - Y[:n].mean(0)
    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y
    num = np.linalg.norm(XtY, 'fro') ** 2
    den = np.linalg.norm(XtX, 'fro') * np.linalg.norm(YtY, 'fro')
    return float(num / den)

cka = linear_cka(X_a0, X_a1)
print(f"  Linear CKA (A0 vs A1): {cka:.4f}")
print(f"  (1.0 = identical geometry, 0.0 = completely different)")

# ── 4. RSA ────────────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("4. RSA (Representational Similarity Analysis)")
print("=" * 50)

n      = min(len(X_a0), len(X_a1), 500)
idx_a0 = np.random.choice(len(X_a0), n, replace=False)
idx_a1 = np.random.choice(len(X_a1), n, replace=False)
rdm_a0 = pdist(X_a0[idx_a0], metric='correlation')
rdm_a1 = pdist(X_a1[idx_a1], metric='correlation')
rsa, p = spearmanr(rdm_a0, rdm_a1)
print(f"  RSA (Spearman r): {rsa:.4f}  (p={p:.2e})")

# ── 5. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#0f0f0f')
for ax in axes.flat:
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='#cccccc')
    ax.xaxis.label.set_color('#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

C_A0 = '#e74c3c'
C_A1 = '#2ecc71'

# plot 1 — linear probe scores
ax = axes[0, 0]
probe_names = list(probe_results.keys())
a0_scores   = [probe_results[k]['a0'] for k in probe_names]
a1_scores   = [probe_results[k]['a1'] for k in probe_names]
x = np.arange(len(probe_names))
w = 0.35
ax.bar(x - w/2, a0_scores, w, label='A0 (Pure BC)',    color=C_A0, alpha=0.85)
ax.bar(x + w/2, a1_scores, w, label='A1 (Predictive)', color=C_A1, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(probe_names, fontsize=10)
ax.set_ylabel('R² / Accuracy', fontsize=11)
ax.set_title('Linear Probes', fontsize=13, fontweight='bold')
ax.legend(facecolor='#2a2a2a', labelcolor='white', fontsize=9)
ax.grid(alpha=0.2, color='#444444', axis='y')
ax.axhline(0, color='white', linewidth=0.5, alpha=0.5)

# plot 2 — singular value spectrum
ax = axes[0, 1]
top_k = 50
ax.plot(range(1, top_k+1), S_a0[:top_k] / S_a0[0], color=C_A0, linewidth=2,
        label=f'A0 (eff_rank={er_a0:.0f})')
ax.plot(range(1, top_k+1), S_a1[:top_k] / S_a1[0], color=C_A1, linewidth=2,
        label=f'A1 (eff_rank={er_a1:.0f})')
ax.set_xlabel('Singular Value Index', fontsize=11)
ax.set_ylabel('Normalised Singular Value', fontsize=11)
ax.set_title('Singular Value Spectrum', fontsize=13, fontweight='bold')
ax.legend(facecolor='#2a2a2a', labelcolor='white', fontsize=9)
ax.grid(alpha=0.2, color='#444444')

# plot 3 — effective rank bar
ax = axes[1, 0]
bars = ax.bar(['A0\n(Pure BC)', 'A1\n(Predictive)'],
              [er_a0, er_a1], color=[C_A0, C_A1], width=0.5)
ax.set_ylabel('Effective Rank', fontsize=11)
ax.set_title('Effective Rank of Encoder Features', fontsize=13, fontweight='bold')
ax.grid(alpha=0.2, color='#444444', axis='y')
for bar, val in zip(bars, [er_a0, er_a1]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}', ha='center', va='bottom', color='white',
            fontweight='bold', fontsize=12)

# plot 4 — CKA and RSA
ax = axes[1, 1]
metrics = ['CKA\n(A0 vs A1)', 'RSA\n(A0 vs A1)']
values  = [cka, rsa]
colors  = ['#3498db', '#9b59b6']
bars    = ax.bar(metrics, values, color=colors, width=0.4, alpha=0.85)
ax.set_ylabel('Similarity Score', fontsize=11)
ax.set_title('Representational Similarity', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1)
ax.grid(alpha=0.2, color='#444444', axis='y')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', color='white',
            fontweight='bold', fontsize=12)

plt.suptitle('Representation Analysis: A0 vs A1 Encoder',
             fontsize=15, fontweight='bold', color='white', y=1.01)
plt.tight_layout()
plt.savefig('representation/representation_analysis.png', dpi=150,
            bbox_inches='tight', facecolor='#0f0f0f')
print()
print("Plot saved to representation/representation_analysis.png")

# ── summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"  CKA between A0 and A1:  {cka:.4f}")
print(f"  RSA between A0 and A1:  {rsa:.4f}")
print(f"  Effective rank — A0: {er_a0:.1f}  A1: {er_a1:.1f}  ({'A1 richer' if er_a1 > er_a0 else 'A0 richer'})")
print(f"  Stable rank    — A0: {sr_a0:.1f}  A1: {sr_a1:.1f}")
print()
for k in probe_results:
    diff = probe_results[k]['a1'] - probe_results[k]['a0']
    print(f"  {k:<15}: A1 {'better' if diff > 0 else 'worse'} by {abs(diff):.3f}")Y