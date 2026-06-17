"""Lightweight BigVul sampling for code feature comparison."""
import pandas as pd
import numpy as np
import os

BIGVUL_PATH = r'E:\paper\new\database\MSR_data_cleaned\MSR_data_cleaned.csv'

print("Reading BigVul sample (first 60k rows)...")
df = pd.read_csv(BIGVUL_PATH, nrows=60000, low_memory=False)
mask_vul = df['vul'] == 1
mask_safe = df['vul'] == 0
print(f"Vul: {mask_vul.sum()}, Safe: {mask_safe.sum()}")

vul_codes = df[mask_vul]['func_before'].dropna().tolist()
safe_codes = df[mask_safe]['func_before'].dropna().sample(n=min(mask_safe.sum(), 3000), random_state=42).tolist()
print(f"Sampled: {len(vul_codes)} vul, {len(safe_codes)} safe")

import pickle
out = {'vul': vul_codes, 'safe': safe_codes, 'vul_total': mask_vul.sum(), 'safe_total': mask_safe.sum()}
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bigvul_sampled.pkl')
with open(outpath, 'wb') as f:
    pickle.dump(out, f)
print(f"Saved: {outpath}")
