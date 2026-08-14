"""
Merge all 320 samples (306 existing + 14 new GLM), recompute GLM-5.1 percentages,
generate Rater 2 blinded CSV.
Convention: human_label=1 means SAFE, human_label=0 means UNSAFE.
"""
import pandas as pd
import numpy as np
import os

rng = np.random.default_rng(42)
base = r'E:\paper\new\SecureProtoCoT\outputs'

# --- 1. Load all existing samples ---
all_samples = []

# DeepSeek (80) - tab-separated, gbk
df = pd.read_csv(os.path.join(base, 'human_eval_blinded.csv'), sep='\t', encoding='gbk')
for _, r in df.iterrows():
    all_samples.append({
        'model': 'DeepSeek-v4-pro', 'config': None,
        'prompt_desc': str(r['prompt_desc']).strip(),
        'code': str(r['code']).strip(),
        'human_label': int(r['human_label']),
    })

# GLM (106 existing) - need to map config from key files
k1 = pd.read_csv(os.path.join(base, 'claude_experiment', 'human_eval_claude_80_key.csv'))
k2 = pd.read_csv(os.path.join(base, 'claude_experiment', 'human_eval_claude_B5B7_key.csv'))
glm_keys = pd.concat([k1, k2])
glm_keys['candidate_id'] = glm_keys['candidate_id'].astype(str)

df_g1 = pd.read_csv(os.path.join(base, 'claude_experiment', 'human_eval_claude_80.csv'))
df_g2 = pd.read_csv(os.path.join(base, 'claude_experiment', 'human_eval_claude_B5B7.csv'))
df_glm = pd.concat([df_g1, df_g2])

# Merge config info
df_glm = df_glm.merge(glm_keys[['eval_id', 'config', 'prompt_id', 'candidate_id']], on='eval_id', how='left')

for _, r in df_glm.iterrows():
    all_samples.append({
        'model': 'GLM-5.1', 'config': r['config'],
        'prompt_desc': str(r['prompt_desc']).strip(),
        'code': str(r['code']).strip(),
        'human_label': int(r['human_label']),
    })

# Sonnet (120)
df_s = pd.read_csv(os.path.join(base, 'sonnet_experiment', 'human_eval_sonnet_120.csv'))
# Sonnet has no config column directly; need to get from mapping
# Actually the Sonnet CSV has eval_id but no config. Let me check if config is needed.
for _, r in df_s.iterrows():
    all_samples.append({
        'model': 'Sonnet-4-6', 'config': None,
        'prompt_desc': str(r['prompt_desc']).strip(),
        'code': str(r['code']).strip(),
        'human_label': int(r['human_label']),
    })

# --- 2. Load 14 new GLM samples ---
df_new = pd.read_csv(os.path.join(base, 'human_eval_glm_new14.csv'))
for _, r in df_new.iterrows():
    all_samples.append({
        'model': 'GLM-5.1', 'config': r['config'],
        'prompt_desc': str(r['prompt_desc']).strip(),
        'code': str(r['code']).strip(),
        'human_label': int(r['human_label']),
    })

print(f'Total samples: {len(all_samples)}')
print(f'  DeepSeek: {sum(1 for s in all_samples if s["model"]=="DeepSeek-v4-pro")}')
print(f'  GLM-5.1: {sum(1 for s in all_samples if s["model"]=="GLM-5.1")}')
print(f'  Sonnet-4-6: {sum(1 for s in all_samples if s["model"]=="Sonnet-4-6")}')

# --- 3. Recompute GLM-5.1 per-config SAFE rates ---
glm_samples = [s for s in all_samples if s['model'] == 'GLM-5.1']
glm_by_cfg = {}
for s in glm_samples:
    cfg = s['config']
    if cfg not in glm_by_cfg:
        glm_by_cfg[cfg] = {'total': 0, 'safe': 0}
    glm_by_cfg[cfg]['total'] += 1
    if s['human_label'] == 1:
        glm_by_cfg[cfg]['safe'] += 1

print('\n=== GLM-5.1 Per-Config SAFE Rates (120 samples) ===')
config_order = ['ISR0', 'ISR1', 'ISR2', 'ISR3', 'B4', 'B5', 'B6', 'B7']
config_paper_names = {
    'ISR0': 'ISR-0', 'ISR1': 'ISR-1', 'ISR2': 'ISR-2', 'ISR3': 'ISR-3',
    'B4': 'SafePrompt', 'B5': 'SVEN', 'B6': 'Reflexion', 'B7': 'CoSec',
}
old_rates = {
    'ISR0': 14.3, 'ISR1': 50.0, 'ISR2': 61.5, 'ISR3': 68.4,
    'B4': 53.8, 'B5': 53.8, 'B6': 46.2, 'B7': 61.5,
}

for cfg in config_order:
    d = glm_by_cfg[cfg]
    rate = d['safe'] / d['total'] * 100
    old = old_rates.get(cfg, 0)
    change = rate - old
    flag = ' *** CHANGED' if abs(change) > 0.5 else ''
    print(f'  {config_paper_names[cfg]:12s} ({cfg:4s}): {d["safe"]:2d}/{d["total"]:2d} = {rate:.1f}%  (was {old:.1f}%, Δ={change:+.1f}pp){flag}')

# --- 4. Check if DeepSeek and Sonnet rates need recomputing ---
# DeepSeek
deepseek_samples = [s for s in all_samples if s['model'] == 'DeepSeek-v4-pro']
ds_safe = sum(1 for s in deepseek_samples if s['human_label'] == 1)
print(f'\nDeepSeek: {ds_safe}/{len(deepseek_samples)} = {ds_safe/len(deepseek_samples)*100:.1f}% SAFE')

# Sonnet
sonnet_samples = [s for s in all_samples if s['model'] == 'Sonnet-4-6']
sn_safe = sum(1 for s in sonnet_samples if s['human_label'] == 1)
print(f'Sonnet: {sn_safe}/{len(sonnet_samples)} = {sn_safe/len(sonnet_samples)*100:.1f}% SAFE')

# --- 5. Generate Rater 2 blinded CSV ---
rng.shuffle(all_samples)

rater2_rows = []
mapping_rows = []
for i, s in enumerate(all_samples, 1):
    rater2_rows.append({
        'eval_id': i,
        'prompt_desc': s['prompt_desc'],
        'code': s['code'],
    })
    mapping_rows.append({
        'new_eval_id': i,
        'model': s['model'],
        'config': s.get('config', ''),
        'rater1_label': s['human_label'],
        'prompt_desc': s['prompt_desc'][:80],
    })

df_r2 = pd.DataFrame(rater2_rows)
out_r2 = os.path.join(base, 'human_eval_rater2_320_blinded.csv')
df_r2.to_csv(out_r2, index=False, encoding='utf-8-sig')
print(f'\nRater 2 blind CSV: {out_r2} ({len(df_r2)} rows)')

df_map = pd.DataFrame(mapping_rows)
out_map = os.path.join(base, 'human_eval_rater2_320_mapping.csv')
df_map.to_csv(out_map, index=False, encoding='utf-8-sig')
print(f'Mapping (KEEP SECRET): {out_map}')

print('\nDone. Send human_eval_rater2_320_blinded.csv to Rater 2.')
print('Rater 2 instructions:')
print('  - Read prompt_desc + code')
print('  - Judge: SAFE (no memory-safety vulnerability) or UNSAFE (at least one confirmed issue)')
print('  - Add columns: human_label (1=SAFE, 0=UNSAFE), issue (description if UNSAFE)')
