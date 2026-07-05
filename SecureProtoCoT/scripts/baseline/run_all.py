#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行所有 baseline 并输出最终对比表
Usage:
  python run_all.py          # 仅打印已完成的baseline对比表
  python run_all.py --run b2 # 运行指定baseline
  python run_all.py --run all # 运行所有baseline
"""
import os, sys, json, csv, importlib
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, load_human_labels

OUTPUT_DIR = Path(CONFIG['output_dir'])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_selection_static(selections, human_labels):
    safe_picks = 0
    total = 0
    for pid, best_idx in selections.items():
        label = human_labels.get((pid, best_idx), -1)
        if label != -1:
            total += 1
            if label == 1:
                safe_picks += 1
    return safe_picks, total


def load_result(baseline_name):
    p = OUTPUT_DIR / baseline_name / 'result.json'
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def compute_b0_b1():
    """从实验B数据计算 B0(Random) 和 B1(Classifier)"""
    import pandas as pd
    import numpy as np

    human_labels = load_human_labels(CONFIG['human_eval_csv'])
    expb_path = Path('/home2/zzl/SecurePath/SecureProtoCoT/outputs/experiment_b')
    df = pd.read_csv(expb_path / 'all_candidates.csv')

    rng = np.random.RandomState(42)

    b0_selections = {}
    b1_selections = {}
    for pid in df['prompt_id'].unique():
        subset = df[df['prompt_id'] == pid]
        b0_selections[pid] = int(subset['candidate_idx'].sample(n=1, random_state=rng).values[0])
        b1_selections[pid] = int(subset.loc[subset['p_vul'].idxmin(), 'candidate_idx'])

    b0_safe, b0_total = evaluate_selection_static(b0_selections, human_labels)
    b1_safe, b1_total = evaluate_selection_static(b1_selections, human_labels)

    return {
        'B0_Random': {'safe_picks': b0_safe, 'total_valid': b0_total,
                       'safe_rate': b0_safe/b0_total*100 if b0_total else 0},
        'B1_Classifier': {'safe_picks': b1_safe, 'total_valid': b1_total,
                           'safe_rate': b1_safe/b1_total*100 if b1_total else 0},
    }


def run_single(baseline_id):
    """运行单个 baseline 脚本"""
    module_map = {
        'b2': 'b2_flawfinder',
        'b3': 'b3_llm_selfeval',
        'b4': 'b4_safe_prompt',
        'b5': 'b5_sven_prompt',
        'b6': 'b6_reflexion',
        'b7': 'b7_cosec_inspired',
    }
    mod = importlib.import_module(module_map[baseline_id])
    return mod.main()


def print_table(results):
    print(f"\n{'='*70}")
    print("BASELINE COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Method':<22} {'Safe/Total':>12} {'Rate':>10}  Notes")
    print("-" * 70)

    notes = {
        'B0_Random': 'Lower bound',
        'B1_Classifier': 'MSR-trained post-hoc (F1=0.896 on MSR)',
        'B2_Flawfinder': 'Static analysis rule engine',
        'B3_LLM_SelfEval': 'LLM self-rating 1-10',
        'B4_SafePrompt': 'Safety-enhanced prompt engineering',
        'B5_SVEN_Prompt': 'SVEN-style safety prefix (prompt adaptation)',
        'B6_Reflexion': 'LLM self-reflection + iterative repair (NeurIPS 2023)',
        'B7_CoSec': 'CoSec co-decoding via prompt (ISSTA 2024, approx)',
    }

    order = ['B0_Random', 'B1_Classifier', 'B2_Flawfinder', 'B3_LLM_SelfEval',
             'B4_SafePrompt', 'B5_SVEN_Prompt', 'B6_Reflexion', 'B7_CoSec']

    for key in order:
        if key in results:
            r = results[key]
            n = notes.get(key, '')
            print(f"{key:<22} {r['safe_picks']:>6}/{r['total_valid']:<5} {r['safe_rate']:>9.1f}%  {n}")
        else:
            print(f"{key:<22} {'(not run yet)':>20}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default=None,
                        choices=['b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'all'],
                        help='Run specified baseline(s)')
    args = parser.parse_args()

    results = {}

    # B0, B1 from experiment_b
    results.update(compute_b0_b1())

    # Run requested baselines
    if args.run:
        targets = ['b2', 'b3', 'b4', 'b5', 'b6', 'b7'] if args.run == 'all' else [args.run]
        for t in targets:
            run_single(t)

    # Collect persisted results
    for dir_name, key in [('b2_flawfinder', 'B2_Flawfinder'),
                           ('b3_llm_selfeval', 'B3_LLM_SelfEval'),
                           ('b4_safe_prompt', 'B4_SafePrompt'),
                           ('b5_sven_prompt', 'B5_SVEN_Prompt'),
                           ('b6_reflexion', 'B6_Reflexion'),
                           ('b7_cosec_inspired', 'B7_CoSec')]:
        r = load_result(dir_name)
        if r:
            results[key] = r

    # Print table
    print_table(results)

    # Save summary
    summary = {k: {'safe_picks': r['safe_picks'], 'total_valid': r['total_valid'],
                   'safe_rate': r['safe_rate']} for k, r in results.items()}
    with open(OUTPUT_DIR / 'baseline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {OUTPUT_DIR / 'baseline_summary.json'}")


if __name__ == '__main__':
    main()
