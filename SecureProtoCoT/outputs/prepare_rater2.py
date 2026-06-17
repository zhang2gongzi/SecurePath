"""
Merge all existing human evaluation samples (320 total) into a single blinded CSV for Rater 2.
Does NOT modify or delete any existing files.
Output: human_eval_rater2_320.csv (blinded, shuffled) + human_eval_rater2_mapping.csv
"""
import pandas as pd
import numpy as np
import os

base = r'E:\paper\new\SecureProtoCoT\outputs'
rng = np.random.default_rng(42)

# --- 1. Read all source files ---
samples = []

# DeepSeek-v4-pro: human_eval_blinded.csv (tab-separated)
fp = os.path.join(base, 'human_eval_blinded.csv')
df = pd.read_csv(fp, sep='\t', encoding='gbk')
print(f'DeepSeek (blinded): {len(df)} rows, cols={list(df.columns)}')
for _, row in df.iterrows():
    samples.append({
        'model': 'DeepSeek-v4-pro',
        'source_file': 'human_eval_blinded.csv',
        'orig_eval_id': row['eval_id'],
        'prompt_desc': str(row['prompt_desc']).strip(),
        'code': str(row['code']).strip(),
        'rater1_label': row['human_label'],
        'rater1_issue': str(row.get('issue', '')).strip(),
    })

# GLM-5.1: human_eval_claude_80.csv (comma-separated)
fp = os.path.join(base, 'claude_experiment', 'human_eval_claude_80.csv')
df = pd.read_csv(fp)
print(f'GLM (claude_80): {len(df)} rows, cols={list(df.columns)}')
for _, row in df.iterrows():
    samples.append({
        'model': 'GLM-5.1',
        'source_file': 'human_eval_claude_80.csv',
        'orig_eval_id': row['eval_id'],
        'prompt_desc': str(row['prompt_desc']).strip(),
        'code': str(row['code']).strip(),
        'rater1_label': row['human_label'],
        'rater1_issue': str(row.get('issue', '')).strip(),
    })

# GLM-5.1: human_eval_claude_B5B7.csv (comma-separated)
fp = os.path.join(base, 'claude_experiment', 'human_eval_claude_B5B7.csv')
df = pd.read_csv(fp)
print(f'GLM (B5B7): {len(df)} rows, cols={list(df.columns)}')
for _, row in df.iterrows():
    samples.append({
        'model': 'GLM-5.1',
        'source_file': 'human_eval_claude_B5B7.csv',
        'orig_eval_id': row['eval_id'],
        'prompt_desc': str(row['prompt_desc']).strip(),
        'code': str(row['code']).strip(),
        'rater1_label': row['human_label'],
        'rater1_issue': str(row.get('issue', '')).strip(),
    })

# Sonnet-4-6: human_eval_sonnet_120.csv (comma-separated)
fp = os.path.join(base, 'sonnet_experiment', 'human_eval_sonnet_120.csv')
df = pd.read_csv(fp)
print(f'Sonnet: {len(df)} rows, cols={list(df.columns)}')
for _, row in df.iterrows():
    samples.append({
        'model': 'Sonnet-4-6',
        'source_file': 'human_eval_sonnet_120.csv',
        'orig_eval_id': row['eval_id'],
        'prompt_desc': str(row['prompt_desc']).strip(),
        'code': str(row['code']).strip(),
        'rater1_label': row['human_label'],
        'rater1_issue': str(row.get('issue', '')).strip(),
    })

print(f'\nTotal samples: {len(samples)}')
print(f'  DeepSeek: {sum(1 for s in samples if s["model"]=="DeepSeek-v4-pro")}')
print(f'  GLM-5.1: {sum(1 for s in samples if s["model"]=="GLM-5.1")}')
print(f'  Sonnet-4-6: {sum(1 for s in samples if s["model"]=="Sonnet-4-6")}')

# --- 2. Shuffle ---
rng.shuffle(samples)

# --- 3. Assign new eval_ids ---
for i, s in enumerate(samples, 1):
    s['new_eval_id'] = i

# --- 4. Build Rater 2 blind CSV (eval_id, prompt_desc, code ONLY) ---
rater2_rows = []
for s in samples:
    rater2_rows.append({
        'eval_id': s['new_eval_id'],
        'prompt_desc': s['prompt_desc'],
        'code': s['code'],
    })

df_rater2 = pd.DataFrame(rater2_rows)
out_rater2 = os.path.join(base, 'human_eval_rater2_320_blinded.csv')
df_rater2.to_csv(out_rater2, index=False)
print(f'\nRater 2 blind CSV: {out_rater2} ({len(df_rater2)} rows)')
print('  Columns: eval_id, prompt_desc, code')
print('  (NO human_label, NO issue, NO model, NO config — fully blinded)')

# --- 5. Build mapping file (for later kappa calculation, DO NOT share with Rater 2) ---
mapping_rows = []
for s in samples:
    mapping_rows.append({
        'new_eval_id': s['new_eval_id'],
        'model': s['model'],
        'source_file': s['source_file'],
        'orig_eval_id': s['orig_eval_id'],
        'prompt_desc': s['prompt_desc'][:80],
        'rater1_label': s['rater1_label'],
    })

df_map = pd.DataFrame(mapping_rows)
out_map = os.path.join(base, 'human_eval_rater2_mapping.csv')
df_map.to_csv(out_map, index=False)
print(f'Mapping file (keep secret): {out_map} ({len(df_map)} rows)')
print('  DO NOT share this file with Rater 2 — it contains Rater 1 labels')

# --- 6. Verify ---
print(f'\n--- Verification ---')
print(f'Rater2 CSV columns: {list(df_rater2.columns)}')
print(f'First 5 eval_ids: {df_rater2["eval_id"].head().tolist()}')
print(f'Sample lengths: {df_rater2["prompt_desc"].nunique()} unique prompts')
print(f'\nDone. Existing files NOT modified.')
