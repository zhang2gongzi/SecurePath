"""
Comprehensive P(vul) analysis across all three models.
- Per-config P(vul) from ALL available candidates (not just eval subset)
- Spearman r from human-evaluated samples
"""
import csv, numpy as np, os
from collections import defaultdict

OUT = os.path.dirname(os.path.abspath(__file__))  # outputs/ directory

def spearman_r(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)+1
    ry = np.argsort(np.argsort(y)).astype(float)+1
    mx, my = np.mean(rx), np.mean(ry)
    return np.sum((rx-mx)*(ry-my))/np.sqrt(np.sum((rx-mx)**2)*np.sum((ry-my)**2))

def per_config_table(scores, label):
    cfg = defaultdict(list)
    for c, vals in scores.items():
        cfg[c].extend(vals)
    print(f'\n{"="*60}')
    print(f'{label}: Per-config P(vul) from ALL candidates')
    print(f'{"Config":<8} {"Mean":>10} {"Median":>10} {"Min":>10} {"Max":>10} {"N":>6}')
    for c in ['ISR0','ISR1','ISR2','ISR3','B4','B5','B6','B7']:
        v = cfg.get(c, [])
        if v:
            print(f'{c:<8} {np.mean(v):>10.6f} {np.median(v):>10.6f} {np.min(v):>10.6f} {np.max(v):>10.6f} {len(v):>6}')

# ====== DeepSeek ======
ds_scores = defaultdict(list)

# ISR data: each CSV has iteration data; use iter=0 for "init", and also track all
for ab in ['ISR-0','ISR-1','ISR-2','ISR-3']:
    path = os.path.join(OUT, f'experiment_isr/iterations_{ab}.csv')
    cfg_name = ab.replace('-','')
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ds_scores[cfg_name].append(float(row['p_vul']))

# Baselines: all_candidates.csv (150 candidates, single run, no config label)
# These are from experiment_b mixed config. Can't separate by baseline.
# Use human_eval_mapping.csv for per-config baseline P(vul) — limited to eval subset
base_path = os.path.join(OUT, 'human_eval_mapping.csv')
with open(base_path, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        src = row['source'].replace('-','').replace('_SVEN','').replace('_COSEC','')
        if src.startswith('ISR'): continue  # ISR handled above
        # Normalize: B4_SAFEPROMPT -> B4, B5_SVEN -> B5, etc.
        cfg = src.split('_')[0] if '_' in src else src
        if cfg in ['B4','B5','B6','B7']:
            ds_scores[cfg].append(float(row['p_vul']))

per_config_table(ds_scores, 'DeepSeek-v4-pro')

# ====== GLM-5.1 ======
glm_scores = defaultdict(list)
with open(os.path.join(OUT, 'claude_experiment/claude_pvul_scores.csv'), 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        glm_scores[row['config']].append(float(row['p_vul']))
per_config_table(glm_scores, 'GLM-5.1')

# ====== Sonnet-4-6 ======
sonnet_scores = defaultdict(list)
with open(os.path.join(OUT, 'sonnet_experiment/sonnet_pvul_scores.csv'), 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        sonnet_scores[row['config']].append(float(row['p_vul']))
per_config_table(sonnet_scores, 'Sonnet-4-6')

# ====== Spearman r ======
print(f'\n{"="*60}')
print('Spearman r: P(vul) vs Human Judgment')

# DeepSeek
pvul_ds, hl_ds = [], []
with open(os.path.join(OUT, 'human_eval_mapping.csv'), 'r', encoding='utf-8-sig') as f:
    ds_map = {int(r['eval_id']): float(r['p_vul']) for r in csv.DictReader(f)}
with open(os.path.join(OUT, 'human_eval_blinded.csv'), 'r', encoding='gbk') as f:
    ds_lbl = {int(r['eval_id']): int(r['human_label']) for r in csv.DictReader(f, delimiter='\t')}
for eid in sorted(set(ds_map)&set(ds_lbl)):
    pvul_ds.append(ds_map[eid]); hl_ds.append(ds_lbl[eid])
print(f'DeepSeek-v4-pro:  r={spearman_r(pvul_ds, hl_ds):.4f}, n={len(pvul_ds)}')

# GLM
def load_gs(data_dir, key_f, lbl_f, pvul_f):
    with open(os.path.join(data_dir,key_f),'r',encoding='utf-8') as f:
        key = {int(r['eval_id']):(r['config'],r['prompt_id'],r['candidate_id']) for r in csv.DictReader(f)}
    with open(os.path.join(data_dir,lbl_f),'r',encoding='utf-8') as f:
        lbl = {int(r['eval_id']):int(r['human_label']) for r in csv.DictReader(f)}
    scores = {}
    with open(os.path.join(data_dir,pvul_f),'r',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            scores[(r['config'],r['prompt_id'],r['candidate_id'])]=float(r['p_vul'])
    pv,hl=[],[]
    for e,(c,pi,ci) in key.items():
        if e in lbl and (c,pi,ci) in scores:
            pv.append(scores[(c,pi,ci)]); hl.append(lbl[e])
    return pv,hl

glm_pv, glm_hl = load_gs(os.path.join(OUT,'claude_experiment'),
    'human_eval_claude_80_key.csv','human_eval_claude_80.csv','claude_pvul_scores.csv')
print(f'GLM-5.1:          r={spearman_r(glm_pv, glm_hl):.4f}, n={len(glm_pv)}')

sonnet_pv, sonnet_hl = load_gs(os.path.join(OUT,'sonnet_experiment'),
    'human_eval_sonnet_120_key.csv','human_eval_sonnet_120.csv','sonnet_pvul_scores.csv')
print(f'Sonnet-4-6:       r={spearman_r(sonnet_pv, sonnet_hl):.4f}, n={len(sonnet_pv)}')

# Combined
all_pv = pvul_ds + glm_pv + sonnet_pv
all_hl = hl_ds + glm_hl + sonnet_hl
print(f'\nAll 280 samples:  r={spearman_r(all_pv, all_hl):.4f}, n={len(all_pv)}')
