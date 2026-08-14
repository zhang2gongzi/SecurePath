"""Compute Cohen's kappa from Rater 2 CSV (handles multiline code fields)."""
import pandas as pd
import numpy as np
import re

base = r'E:\paper\new\SecureProtoCoT\outputs'

# Read raw file
with open(f'{base}/human_eval_rater2_320_blinded.csv', 'r', encoding='utf-8-sig') as f:
    text = f.read()

# Split into records: each starts with eval_id at beginning of line
# Pattern: ^\d+,  (eval_id followed by comma)
records = re.split(r'\n(?=\d+,)', text)

# Skip header
if records[0].startswith('eval_id'):
    records = records[1:]

print(f'Found {len(records)} records')

# Extract human_label from each record
r2_labels = {}
for rec in records:
    # Find eval_id: first number
    m = re.match(r'(\d+),', rec)
    if not m:
        continue
    eid = int(m.group(1))

    # Find human_label: look for the pattern ",0," or ",1," near end
    # The format is: eval_id,prompt_desc,code,human_label,issue
    # Where issue may have commas but human_label is always 0 or 1
    # Strategy: find last occurrence of ",1," or ",0," followed by rest
    # Actually simpler: find the second-to-last field by looking for ,0, or ,1,
    # then everything after is issue

    # The human_label is the last single-digit field
    # Match: ,0,<rest> or ,1,<rest> or ,0\n or ,1\n at end
    # We need to find which ,0 or ,1 is the actual label (not part of code/issue)
    # Heuristic: look for ",0," or ",1," that appears within last 200 chars
    tail = rec[-300:] if len(rec) > 300 else rec
    # Pattern: comma, digit 0/1, comma or end
    matches = list(re.finditer(r',([01])(?:,|$)', tail))
    if matches:
        # The human_label is likely the last match before the issue field
        # Take the match closest to end
        label = int(matches[-1].group(1))
        r2_labels[eid] = label

print(f'Parsed {len(r2_labels)} labels')

# Load mapping
mapping = pd.read_csv(f'{base}/human_eval_rater2_320_mapping.csv')

# Align
r1_vals, r2_vals = [], []
for _, row in mapping.iterrows():
    eid = row['new_eval_id']
    if eid in r2_labels:
        r1_vals.append(row['rater1_label'])
        r2_vals.append(r2_labels[eid])

r1 = np.array(r1_vals)
r2 = np.array(r2_vals)
print(f'Matched pairs: {len(r1)}')

# Cohen's kappa
n = len(r1)
n00 = ((r1==0)&(r2==0)).sum()
n01 = ((r1==0)&(r2==1)).sum()
n10 = ((r1==1)&(r2==0)).sum()
n11 = ((r1==1)&(r2==1)).sum()
po = (n00 + n11) / n
pe = ((n00+n01)*(n00+n10) + (n10+n11)*(n01+n11)) / (n*n)
kappa = (po - pe) / (1.0 - pe) if pe < 1.0 else 1.0

print(f'\n=== Cohen kappa = {kappa:.4f} ===')
print(f'Agreement: {po*100:.1f}% ({n00+n11}/{n})')
print(f'R1=SAFE  R2=SAFE:  {n11}')
print(f'R1=SAFE  R2=UNSAFE:{n10}')
print(f'R1=UNSAFE R2=SAFE: {n01}')
print(f'R1=UNSAFE R2=UNSAFE:{n00}')

# Interpretation
if kappa < 0: qual = 'poor'
elif kappa < 0.2: qual = 'slight'
elif kappa < 0.4: qual = 'fair'
elif kappa < 0.6: qual = 'moderate'
elif kappa < 0.8: qual = 'substantial'
else: qual = 'almost perfect'
print(f'Interpretation: {qual} agreement')

# Per-model
merged = mapping.copy()
merged['r2_label'] = merged['new_eval_id'].map(r2_labels)
merged = merged.dropna(subset=['r2_label'])
merged['r2_label'] = merged['r2_label'].astype(int)

for model in ['DeepSeek-v4-pro', 'GLM-5.1', 'Sonnet-4-6']:
    sub = merged[merged['model'] == model]
    k = cohen_kappa(sub['rater1_label'].values, sub['r2_label'].values)
    print(f'{model}: kappa={k:.4f} (n={len(sub)})')

def cohen_kappa(r1, r2):
    n = len(r1)
    n00 = ((r1==0)&(r2==0)).sum()
    n01 = ((r1==0)&(r2==1)).sum()
    n10 = ((r1==1)&(r2==0)).sum()
    n11 = ((r1==1)&(r2==1)).sum()
    po = (n00 + n11) / n
    pe = ((n00+n01)*(n00+n10) + (n10+n11)*(n01+n11)) / (n*n)
    return (po - pe) / (1.0 - pe) if pe < 1.0 else 1.0
