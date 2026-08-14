"""
Fig 3: Cross-domain distribution shift — P(vul) vs human judgment across three models.
Three-panel scatter plot with loess trend lines and Spearman r annotations.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import csv
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(BASE), 'outputs')

def load_deepseek():
    """DeepSeek: match human_eval_mapping.csv (p_vul) + human_eval_blinded.csv (human_label) by eval_id."""
    pvul = {}
    with open(os.path.join(OUT_DIR, 'human_eval_mapping.csv'), 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            pvul[int(row['eval_id'])] = float(row['p_vul'])
    labels = {}
    with open(os.path.join(OUT_DIR, 'human_eval_blinded.csv'), 'r', encoding='gbk') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            labels[int(row['eval_id'])] = int(row['human_label'])
    pv, hl = [], []
    for eid in sorted(set(pvul) & set(labels)):
        pv.append(pvul[eid])
        hl.append(labels[eid])
    return np.array(pv), np.array(hl)

def load_glm_sonnet(model_name):
    """GLM or Sonnet: match key file + labeled CSV + pvul scores."""
    if model_name == 'GLM-5.1':
        data_dir = os.path.join(OUT_DIR, 'claude_experiment')
        key_file = 'human_eval_claude_80_key.csv'
        label_file = 'human_eval_claude_80.csv'
        pvul_file = 'claude_pvul_scores.csv'
    else:
        data_dir = os.path.join(OUT_DIR, 'sonnet_experiment')
        key_file = 'human_eval_sonnet_120_key.csv'
        label_file = 'human_eval_sonnet_120.csv'
        pvul_file = 'sonnet_pvul_scores.csv'

    # Load key
    key = {}
    with open(os.path.join(data_dir, key_file), 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key[int(row['eval_id'])] = (row['config'], row['prompt_id'], row['candidate_id'])
    # Load labels
    labels = {}
    with open(os.path.join(data_dir, label_file), 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            labels[int(row['eval_id'])] = int(row['human_label'])
    # Load scores
    scores = {}
    with open(os.path.join(data_dir, pvul_file), 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            scores[(row['config'], row['prompt_id'], row['candidate_id'])] = float(row['p_vul'])

    pv, hl = [], []
    for eid, (cfg, pid, cid) in key.items():
        if eid not in labels:
            continue
        sk = (cfg, pid, cid)
        if sk in scores:
            pv.append(scores[sk])
            hl.append(labels[eid])
    return np.array(pv), np.array(hl)

def spearman_r(x, y):
    """Manual Spearman rank correlation."""
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(float) + 1
    ry = np.argsort(np.argsort(y)).astype(float) + 1
    mx, my = np.mean(rx), np.mean(ry)
    num = np.sum((rx - mx) * (ry - my))
    den = np.sqrt(np.sum((rx - mx)**2) * np.sum((ry - my)**2))
    return num / den if den > 0 else 0.0

# --- Load data ---
ds_pv, ds_hl = load_deepseek()
glm_pv, glm_hl = load_glm_sonnet('GLM-5.1')
sonnet_pv, sonnet_hl = load_glm_sonnet('Sonnet-4-6')

print(f"DeepSeek: n={len(ds_pv)}, r={spearman_r(ds_pv, ds_hl):.3f}")
print(f"GLM-5.1:  n={len(glm_pv)}, r={spearman_r(glm_pv, glm_hl):.3f}")
print(f"Sonnet:   n={len(sonnet_pv)}, r={spearman_r(sonnet_pv, sonnet_hl):.3f}")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
models = [
    ('DeepSeek-v4-pro', ds_pv, ds_hl, '#4472C4'),
    ('GLM-5.1', glm_pv, glm_hl, '#ED7D31'),
    ('Sonnet-4-6', sonnet_pv, sonnet_hl, '#70AD47'),
]

for ax, (name, pv, hl, color) in zip(axes, models):
    safe_mask = hl == 1
    unsafe_mask = hl == 0

    # Jitter Y for visibility
    rng = np.random.RandomState(42)
    y_jitter = rng.uniform(-0.04, 0.04, len(hl))

    # Plot points
    ax.scatter(pv[unsafe_mask], 0.0 + y_jitter[unsafe_mask],
               c='#D62728', s=28, alpha=0.5, label='UNSAFE', marker='x', linewidths=0.8)
    ax.scatter(pv[safe_mask], 1.0 + y_jitter[safe_mask],
               c='#2CA02C', s=28, alpha=0.5, label='SAFE', marker='o', edgecolors='none')

    # Loess-style trend: bin means
    bins = np.concatenate([[0], np.logspace(-4.5, 0, 15)])
    bin_centers, bin_means = [], []
    for i in range(len(bins)-1):
        mask = (pv >= bins[i]) & (pv < bins[i+1])
        if mask.sum() >= 3:
            bin_centers.append(np.sqrt(bins[i]*bins[i+1]))
            bin_means.append(hl[mask].mean())
    if len(bin_centers) > 1:
        ax.plot(bin_centers, bin_means, '-', color=color, linewidth=2, alpha=0.8)

    # Spearman r
    r_val = spearman_r(pv, hl)
    ax.text(0.95, 0.08, rf"Spearman $\rho$ = {r_val:.2f}", transform=ax.transAxes,
            fontsize=11, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor=color))

    ax.set_title(name, fontsize=12, fontweight='bold', color=color)
    ax.set_xscale('symlog', linthresh=1e-4)
    ax.set_xlabel('Classifier P(vul)', fontsize=9.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['UNSAFE\n(0)', 'SAFE\n(1)'], fontsize=9)
    ax.set_ylim(-0.2, 1.2)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    ax.grid(axis='y', alpha=0.2)

# Shared styling
axes[0].set_ylabel('Human Judgment', fontsize=10)
fig.suptitle('Cross-Domain Distribution Shift: Classifier P(vul) vs. Human Safety Judgment',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(BASE, 'fig3_domain_shift.pdf'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(BASE, 'fig3_domain_shift.png'), dpi=200, bbox_inches='tight')
print("\nSaved: fig3_domain_shift.pdf / .png")
