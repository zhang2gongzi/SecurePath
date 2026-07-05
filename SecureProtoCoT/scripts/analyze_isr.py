#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析 ISR 实验结果：各消融配置对比 + 人工标注验证
"""
import csv, json, os, numpy as np, pandas as pd
from pathlib import Path

OUTPUT_DIR = '/home2/zzl/SecurePath/SecureProtoCoT/outputs/experiment_isr'
HUMAN_EVAL_PATH = '/home2/zzl/SecurePath/SecureProtoCoT/outputs/experiment_b/human_eval.csv'

ABLATION_LABELS = {
    'ISR-0': 'No Feedback (Baseline)',
    'ISR-1': 'Generic Feedback',
    'ISR-2': 'Attention-Guided Feedback',
    'ISR-3': 'Attention-Guided + Safety Spec',
}


def load_human_labels(path):
    labels = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            labels[(row['prompt_id'], int(row['candidate_idx']))] = int(row['human_label'])
    return labels


def main():
    labels = load_human_labels(HUMAN_EVAL_PATH)
    print(f"Loaded {len(labels)} human labels")

    print(f"\n{'='*80}")
    print("ISR 实验分析：迭代安全精炼效果")
    print(f"{'='*80}")

    for ablation_id in ['ISR-0', 'ISR-1', 'ISR-2', 'ISR-3']:
        path = os.path.join(OUTPUT_DIR, f'iterations_{ablation_id}.csv')
        if not os.path.exists(path):
            print(f"\n{ablation_id}: NO DATA")
            continue

        df = pd.read_csv(path)
        print(f"\n{'='*60}")
        print(f"{ablation_id}: {ABLATION_LABELS.get(ablation_id, ablation_id)}")
        print(f"{'='*60}")

        prompts = df['prompt_id'].unique()
        print(f"Prompts: {len(prompts)}, Total iterations: {len(df)}")

        init_pvuls = []
        best_pvuls = []
        reductions = []
        converged = 0
        prompt_details = []

        for pid in prompts:
            subset = df[df['prompt_id'] == pid]
            init = subset[subset['iteration'] == subset['iteration'].min()]['p_vul'].values[0]
            best = subset['p_vul'].min()
            best_iter = subset[subset['p_vul'].idxmin()]['iteration']
            num_iters = subset['iteration'].max() + 1

            init_pvuls.append(init)
            best_pvuls.append(best)
            reductions.append(init - best)
            if best < 0.001:
                converged += 1

            prompt_details.append({
                'prompt_id': pid,
                'initial_p_vul': init,
                'best_p_vul': best,
                'best_iteration': best_iter,
                'num_iterations': num_iters,
            })

        print(f"\nP(vul) Analysis:")
        print(f"  Initial:   mean={np.mean(init_pvuls):.6f}, median={np.median(init_pvuls):.6f}")
        print(f"  Best:      mean={np.mean(best_pvuls):.6f}, median={np.median(best_pvuls):.6f}")
        print(f"  Reduction: mean={np.mean(reductions):.6f}, median={np.median(reductions):.6f}")
        print(f"  Converged (P(vul)<0.001): {converged}/{len(prompts)}")

        top3_improved = sorted(prompt_details, key=lambda x: x['initial_p_vul'] - x['best_p_vul'], reverse=True)[:3]
        print(f"\n  Top 3 improvements:")
        for d in top3_improved:
            delta = d['initial_p_vul'] - d['best_p_vul']
            print(f"    {d['prompt_id']}: {d['initial_p_vul']:.6f} → {d['best_p_vul']:.6f} "
                  f"(Δ={delta:+.6f}, {int(d['num_iterations'])} iters)")

        # P(vul) trajectory
        for d in prompt_details[:3]:
            pid = d['prompt_id']
            subset = df[df['prompt_id'] == pid].sort_values('iteration')
            traj = ' → '.join([f"{r['p_vul']:.6f}" for _, r in subset.iterrows()])
            print(f"\n  {pid} trajectory: {traj}")

    # Cross-ablation comparison
    print(f"\n{'='*80}")
    print("CROSS-ABLATION COMPARISON")
    print(f"{'='*80}")
    print(f"{'Ablation':<10} {'Init P(vul)':>12} {'Best P(vul)':>12} {'Δ P(vul)':>12} "
          f"{'Converged':>10} {'Avg Iters':>10}")
    print("-" * 68)

    for ablation_id in ['ISR-0', 'ISR-1', 'ISR-2', 'ISR-3']:
        path = os.path.join(OUTPUT_DIR, f'iterations_{ablation_id}.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        init_vals = df.groupby('prompt_id')['p_vul'].first()
        best_vals = df.groupby('prompt_id')['p_vul'].min()
        num_iters = df.groupby('prompt_id')['iteration'].max() + 1
        reductions = init_vals - best_vals
        conv = (best_vals < 0.001).sum()

        print(f"{ablation_id:<10} {init_vals.mean():>12.6f} {best_vals.mean():>12.6f} "
              f"{reductions.mean():>+12.6f} {conv:>8}/{len(best_vals)} {num_iters.mean():>10.1f}")

    print(f"\n{'='*80}")
    print("Key Findings:")
    print(f"{'='*80}")
    print("1. Does P(vul) decrease with iterative feedback? (ISR-1/2/3 vs ISR-0)")
    print("2. Does attention-guided feedback outperform generic feedback? (ISR-2 vs ISR-1)")
    print("3. Does safety spec add value beyond attention feedback? (ISR-3 vs ISR-2)")
    print("4. How many iterations are needed for convergence?")
    print("5. Are there cases where feedback makes code WORSE (P(vul) increases)?")


if __name__ == '__main__':
    main()
