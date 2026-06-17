#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Post-hoc P(vul) scoring for ISR-coinflip experiment.
Run on server with classifier models available.

Input:  experiment_isr_coinflip output (iterations_coinflip.csv)
Output: iterations_coinflip_scored.csv (with p_vul column added)
        final_per_run_scored.csv (final code per run, with p_vul)
        best_per_prompt.csv (best code per prompt by p_vul, for human eval)
"""
import os, sys, torch, numpy as np, pandas as pd
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from attention_pooling import SafetyClassifier

CONFIG = {
    'encoder_path': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/models/best_model',
    'classifier_path': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/models/safety_classifier_attn.pt',
    'data_dir': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/experiment_isr_coinflip',
    'max_length': 512,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


def load_models():
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['encoder_path'])
    codebert = AutoModel.from_pretrained(CONFIG['encoder_path']).to(CONFIG['device'])
    codebert.eval()
    classifier = SafetyClassifier().to(CONFIG['device'])
    classifier.load_state_dict(torch.load(CONFIG['classifier_path'], map_location=CONFIG['device']))
    classifier.eval()
    return tokenizer, codebert, classifier


@torch.no_grad()
def score_code(code, tokenizer, codebert, classifier):
    device = CONFIG['device']
    enc = tokenizer(code, max_length=CONFIG['max_length'], padding='max_length',
                    truncation=True, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    outputs = codebert(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = outputs.last_hidden_state
    logits = classifier(hidden_states, attention_mask)
    p_vul = torch.nn.functional.softmax(logits, dim=1)[0, 1].item()
    return p_vul


def main():
    data_dir = Path(CONFIG['data_dir'])
    csv_path = data_dir / 'iterations_coinflip.csv'

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run experiment_isr_coinflip.py first.")
        sys.exit(1)

    print("Loading models...")
    tokenizer, codebert, classifier = load_models()
    print(f"Models loaded on {CONFIG['device']}")

    df = pd.read_csv(csv_path)
    print(f"Scoring {len(df)} code samples...")

    p_vuls = []
    for i, row in df.iterrows():
        p_vul = score_code(str(row['code']), tokenizer, codebert, classifier)
        p_vuls.append(p_vul)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(df)} done")

    df['p_vul'] = p_vuls
    scored_path = data_dir / 'iterations_coinflip_scored.csv'
    df.to_csv(scored_path, index=False)
    print(f"Saved: {scored_path}")

    # Build final_per_run_scored: last iteration of each run
    final_rows = []
    for (pid, rid), group in df.groupby(['prompt_id', 'run_id']):
        last = group.loc[group['iteration'].idxmax()]
        final_rows.append({
            'prompt_id': pid,
            'run_id': rid,
            'num_iterations': int(last['iteration']) + 1,
            'p_vul': last['p_vul'],
            'code': last['code'],
        })
    df_final = pd.DataFrame(final_rows)
    df_final.to_csv(data_dir / 'final_per_run_scored.csv', index=False)
    print(f"Saved: {data_dir / 'final_per_run_scored.csv'}")

    # Build best_per_prompt: pick run with lowest p_vul per prompt
    best_rows = []
    for pid, group in df_final.groupby('prompt_id'):
        best = group.loc[group['p_vul'].idxmin()]
        best_rows.append(best.to_dict())
    df_best = pd.DataFrame(best_rows)
    df_best.to_csv(data_dir / 'best_per_prompt.csv', index=False)
    print(f"Saved: {data_dir / 'best_per_prompt.csv'}")

    # Summary
    print(f"\n{'='*60}")
    print("SCORING SUMMARY")
    print(f"{'='*60}")
    print(f"  Samples scored:    {len(df)}")
    print(f"  Avg P(vul):        {np.mean(p_vuls):.6f}")
    print(f"  Median P(vul):     {np.median(p_vuls):.6f}")
    final_pvuls = df_final['p_vul'].values
    print(f"  Final-code avg:    {np.mean(final_pvuls):.6f}")
    print(f"  Best-per-prompt:   {df_best['p_vul'].mean():.6f}")
    print(f"\n  ISR-1 reference:   avg_best=0.031")


if __name__ == '__main__':
    main()
