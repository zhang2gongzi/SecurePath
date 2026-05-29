#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B2: Flawfinder 静态分析 baseline
对 experiment_b 的 150 条候选代码跑 Flawfinder，选告警最少的代码
"""
import os, sys, subprocess, tempfile, csv, json, re
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, load_human_labels, evaluate_selection

OUTPUT_DIR = Path(CONFIG['output_dir']) / 'b2_flawfinder'


def run_flawfinder(code):
    """对一段代码跑 flawfinder，返回 (total_warnings, warning_details)"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False, encoding='utf-8') as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ['flawfinder', '--quiet', tmp_path],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout + result.stderr
    except FileNotFoundError:
        print("  WARNING: flawfinder not installed. pip install flawfinder")
        return 0, []
    except subprocess.TimeoutExpired:
        return 0, []
    finally:
        os.unlink(tmp_path)

    warnings = []
    for line in output.split('\n'):
        if ':' in line and not line.startswith('Flawfinder') and not line.startswith('Checking'):
            parts = line.split(':')
            if len(parts) >= 2:
                try:
                    level_str = parts[0].strip()
                    level = int(level_str) if level_str.isdigit() else 0
                    warnings.append(level)
                except ValueError:
                    continue

    return len(warnings), warnings


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    human_labels = load_human_labels(CONFIG['human_eval_csv'])

    print("=" * 60)
    print("B2: Flawfinder Static Analysis Baseline")
    print("=" * 60)

    df = pd.read_csv(CONFIG['all_candidates_csv'])
    print(f"Loaded {len(df)} candidates from {len(df['prompt_id'].unique())} prompts")

    results = []
    per_prompt = defaultdict(list)

    for _, row in df.iterrows():
        pid = row['prompt_id']
        cid = int(row['candidate_idx'])
        code = str(row['code'])
        p_vul = row['p_vul']

        n_warnings, warning_levels = run_flawfinder(code)
        per_prompt[pid].append((cid, n_warnings, p_vul, code))
        results.append({
            'prompt_id': pid, 'candidate_idx': cid,
            'flawfinder_warnings': n_warnings,
            'classifier_p_vul': p_vul,
        })

        if (len(results)) % 20 == 0:
            print(f"  Processed {len(results)}/150...")

    pd.DataFrame(results).to_csv(OUTPUT_DIR / 'flawfinder_scores.csv', index=False)

    selections = {}
    for pid, candidates in per_prompt.items():
        candidates.sort(key=lambda x: x[1])
        best = candidates[0]
        selections[pid] = best[0]
        n_w = best[1]
        human_label = human_labels.get((pid, best[0]), -1)
        label_str = "SAFE" if human_label == 1 else ("UNSAFE" if human_label == 0 else "?")
        print(f"  {pid}: c{best[0]} ({n_w} warnings) [{label_str}]")

    safe_picks, total, misses = evaluate_selection(selections, human_labels)
    rate = safe_picks / total * 100 if total > 0 else 0

    print(f"\nB2 Result: {safe_picks}/{total} safe ({rate:.1f}%)")
    if misses:
        print("Misses:")
        for pid, idx, safe_alt in misses:
            print(f"  {pid} c{idx}: {safe_alt} safe alternatives available")

    with open(OUTPUT_DIR / 'result.json', 'w') as f:
        json.dump({
            'baseline': 'B2_Flawfinder',
            'safe_picks': safe_picks, 'total_valid': total, 'safe_rate': rate,
            'selections': {k: int(v) for k, v in selections.items()},
            'misses': [{'prompt_id': p, 'selected_idx': i, 'safe_alternatives': s} for p, i, s in misses],
        }, f, indent=2)

    return safe_picks, total, rate


if __name__ == '__main__':
    main()
