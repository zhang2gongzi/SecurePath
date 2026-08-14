"""
Analyze P(vul) scores for GLM and Sonnet experiments.
- Per-config P(vul) summary
- Spearman correlation with human evaluation
"""
import csv
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
import os

BASE = os.path.dirname(os.path.abspath(__file__))

def load_scores(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['config'], row['prompt_id'], row['candidate_id'])
            data[key] = float(row['p_vul'])
    return data

def per_config_summary(scores, label):
    cfg = defaultdict(list)
    for (c, p, _), v in scores.items():
        cfg[c].append(v)
    print(f"\n=== {label}: Per-config P(vul) ===")
    print(f"{'Config':<8} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'N':>6}")
    for c in ['ISR0','ISR1','ISR2','ISR3','B4','B5','B6','B7']:
        vals = cfg[c]
        print(f"{c:<8} {np.mean(vals):>10.6f} {np.median(vals):>10.6f} {np.min(vals):>10.6f} {np.max(vals):>10.6f} {len(vals):>6}")

def load_human_eval(key_path, label_path):
    """Load human eval labels from key file + labeled CSV."""
    # Key: eval_id -> (config, prompt_id, candidate_id)
    key = {}
    with open(key_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key[int(row['eval_id'])] = (row['config'], row['prompt_id'], row['candidate_id'])

    # Labels: eval_id -> label (1=SAFE, 0=UNSAFE)
    labels = {}
    with open(label_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[int(row['eval_id'])] = int(row['human_label'])

    return key, labels

def compute_spearman(key, labels, scores, label_name):
    """Compute Spearman r between P(vul) and human judgment."""
    pvul_list = []
    human_list = []
    for eval_id, (config, prompt_id, cand_id) in key.items():
        if eval_id not in labels:
            continue
        score_key = (config, prompt_id, cand_id)
        if score_key in scores:
            pvul_list.append(scores[score_key])
            human_list.append(labels[eval_id])  # 1=SAFE, 0=UNSAFE

    if len(pvul_list) < 5:
        print(f"\n=== {label_name}: Not enough matched samples ({len(pvul_list)}) ===")
        return

    r, p = spearmanr(pvul_list, human_list)
    print(f"\n=== {label_name}: Spearman r={r:.4f}, p={p:.4f}, n={len(pvul_list)} ===")

    # P(vul) by human label
    safe_pvul = [pv for pv, h in zip(pvul_list, human_list) if h == 1]
    unsafe_pvul = [pv for pv, h in zip(pvul_list, human_list) if h == 0]
    print(f"  Human SAFE:   P(vul) mean={np.mean(safe_pvul):.6f}, median={np.median(safe_pvul):.6f}")
    print(f"  Human UNSAFE: P(vul) mean={np.mean(unsafe_pvul):.6f}, median={np.median(unsafe_pvul):.6f}")

    # P(vul) band analysis
    bands = [(0, 0.001), (0.001, 0.01), (0.01, 0.1), (0.1, 1.0)]
    print(f"  {'P(vul) Band':<20} {'N':>5} {'Human SAFE%':>12}")
    for lo, hi in bands:
        band_data = [(pv, h) for pv, h in zip(pvul_list, human_list) if lo <= pv < hi]
        if band_data:
            safe_pct = sum(h for _, h in band_data) / len(band_data) * 100
            print(f"  [{lo:.4f}, {hi:.4f})  {len(band_data):>5}  {safe_pct:>10.1f}%")

# Load scores
glm_scores = load_scores(os.path.join(BASE, 'claude_pvul_scores.csv'))
sonnet_scores = load_scores(os.path.join(BASE, 'sonnet_pvul_scores.csv'))

# Per-config summary
per_config_summary(glm_scores, 'GLM-5.1')
per_config_summary(sonnet_scores, 'Sonnet-4-6')

# Human eval correlation
# GLM: 80 samples
glm_key, glm_labels = load_human_eval(
    os.path.join(BASE, 'human_eval_claude_80_key.csv'),
    os.path.join(BASE, 'human_eval_claude_80.csv'))
compute_spearman(glm_key, glm_labels, glm_scores, 'GLM-5.1 (80 samples)')

# GLM B5/B7 extra: 26 samples
glm_b5b7_key, glm_b5b7_labels = load_human_eval(
    os.path.join(BASE, 'human_eval_claude_B5B7_key.csv'),
    os.path.join(BASE, 'human_eval_claude_B5B7.csv'))
compute_spearman(glm_b5b7_key, glm_b5b7_labels, glm_scores, 'GLM-5.1 B5B7 (26 samples)')

# Sonnet: 120 samples
sonnet_key, sonnet_labels = load_human_eval(
    os.path.join(BASE, 'human_eval_sonnet_120_key.csv'),
    os.path.join(BASE, 'human_eval_sonnet_120.csv'))
compute_spearman(sonnet_key, sonnet_labels, sonnet_scores, 'Sonnet-4-6 (120 samples)')
