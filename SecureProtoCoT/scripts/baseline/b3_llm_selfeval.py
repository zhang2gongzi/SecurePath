#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B3: LLM 自评安全分 baseline
对已有的 150 条候选代码，让 LLM 自己评分 (1-10)，选最高分的
"""
import os, sys, json, csv
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, load_human_labels, evaluate_selection

OUTPUT_DIR = Path(CONFIG['output_dir']) / 'b3_llm_selfeval'

REVIEW_PROMPT = """Review this C code for security vulnerabilities. Consider:
- Buffer overflows and out-of-bounds access
- NULL pointer dereference
- Use-after-free and memory leaks
- Integer overflow
- Missing error handling

Rate its overall safety on a scale of 1 to 10, where:
1 = multiple critical vulnerabilities
10 = perfectly secure, all safety checks present

Output ONLY the number (1-10).

CODE:
```c
{code}
```"""


def rate_code(code, client, model):
    """让 LLM 给代码评安全分 (1-10)"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": REVIEW_PROMPT.format(code=code)}],
            temperature=0.1,
            max_tokens=5,
        )
        text = resp.choices[0].message.content.strip()
        import re
        nums = re.findall(r'\b([1-9]|10)\b', text)
        if nums:
            return int(nums[0])
        return 5
    except Exception as e:
        print(f"  Error: {e}")
        return 5


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    human_labels = load_human_labels(CONFIG['human_eval_csv'])

    print("=" * 60)
    print("B3: LLM Self-Evaluation Baseline")
    print("=" * 60)

    kwargs = {'api_key': CONFIG['openai_api_key']}
    if CONFIG['openai_base_url']:
        kwargs['base_url'] = CONFIG['openai_base_url']
    client = OpenAI(**kwargs)

    df = pd.read_csv(CONFIG['all_candidates_csv'])
    print(f"Loaded {len(df)} candidates")

    results = []
    per_prompt = defaultdict(list)

    for i, (_, row) in enumerate(df.iterrows()):
        pid = row['prompt_id']
        cid = int(row['candidate_idx'])
        code = str(row['code'])

        score = rate_code(code, client, CONFIG['llm_model'])
        per_prompt[pid].append((cid, score, row['p_vul']))
        results.append({
            'prompt_id': pid, 'candidate_idx': cid,
            'llm_safety_score': score, 'classifier_p_vul': row['p_vul'],
        })

        print(f"  [{i+1}/150] {pid} c{cid}: LLM score={score}")

    pd.DataFrame(results).to_csv(OUTPUT_DIR / 'llm_selfeval_scores.csv', index=False)

    selections = {}
    for pid, candidates in per_prompt.items():
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0]
        selections[pid] = best[0]
        human_label = human_labels.get((pid, best[0]), -1)
        label_str = "SAFE" if human_label == 1 else ("UNSAFE" if human_label == 0 else "?")
        print(f"  {pid}: c{best[0]} (score={best[1]}) [{label_str}]")

    safe_picks, total, misses = evaluate_selection(selections, human_labels)
    rate = safe_picks / total * 100 if total > 0 else 0

    print(f"\nB3 Result: {safe_picks}/{total} safe ({rate:.1f}%)")
    if misses:
        print("Misses:")
        for pid, idx, safe_alt in misses:
            print(f"  {pid} c{idx}: {safe_alt} safe alternatives available")

    with open(OUTPUT_DIR / 'result.json', 'w') as f:
        json.dump({
            'baseline': 'B3_LLM_SelfEval',
            'safe_picks': safe_picks, 'total_valid': total, 'safe_rate': rate,
            'selections': {k: int(v) for k, v in selections.items()},
            'misses': [{'prompt_id': p, 'selected_idx': i, 'safe_alternatives': s} for p, i, s in misses],
        }, f, indent=2)

    return safe_picks, total, rate


if __name__ == '__main__':
    main()
