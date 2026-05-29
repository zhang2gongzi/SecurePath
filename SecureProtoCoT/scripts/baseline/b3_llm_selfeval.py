#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B3: LLM 自评安全分 baseline
对已有的 150 条候选代码，让 LLM 自己评分 (1-10)，选最高分的
"""
import os, sys, json, csv, re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, load_human_labels, evaluate_selection

OUTPUT_DIR = Path(CONFIG['output_dir']) / 'b3_llm_selfeval'

REVIEW_PROMPT = """You are an expert C/C++ security auditor. 
Review the following code for critical vulnerabilities (buffer overflow, null dereference, UAF, integer overflow, missing checks).
Output ONLY a single integer from 1 to 10 representing its safety level.
1 = critically unsafe, 10 = perfectly secure.
Do not output any explanation or markdown.

CODE:
```c
{code}
```"""


def rate_code(code, client, model):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": REVIEW_PROMPT.format(code=code)}],
            temperature=0.1,
            max_tokens=20,
        )
        text = resp.choices[0].message.content.strip()
        print(f"DEBUG: Raw response: '{text}'")
        
        import re
        nums = re.findall(r'\b([1-9]|10)\b', text)
        if nums:
            return int(nums[0])
        print(f"DEBUG: No number found, returning default 5")
        return 5
    except Exception as e:
        print(f"DEBUG: API Error: {e}")
        return 5


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    human_labels = load_human_labels(CONFIG['human_eval_csv'])

    print("=" * 60)
    print("B3: LLM Self-Evaluation Baseline")
    print("=" * 60)

    # 🔑 已按你的要求直接硬编码 API Key 和地址
    # ⚠️ 注意：DeepSeek 的 OpenAI 兼容接口必须带 /v1 后缀，否则网关会报 Authentication Fails
    API_KEY = "sk-43433e3f388b48498a3d9f5669cd42a2"  # ← 如失效/过期，请在此替换新 Key
    BASE_URL = "https://api.deepseek.com"             # ← 必须保留 /v1

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

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