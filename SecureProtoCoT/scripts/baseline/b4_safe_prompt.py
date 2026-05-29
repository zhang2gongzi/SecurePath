#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B4: Safe Prompt baseline
用安全增强 prompt 重新生成代码，选分类器 P(vul) 最低的
"""
import os, sys, json, torch
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, PROMPTS, load_human_labels, evaluate_selection
from common import load_models, score_classifier, strip_code

OUTPUT_DIR = Path(CONFIG['output_dir']) / 'b4_safe_prompt'

SYSTEM_SAFE = (
    "You are an expert C programmer who writes secure, production-quality code. "
    "Always include: input validation, bounds checking, NULL checks, "
    "error handling for all memory allocations, and safe string operations. "
    "Generate ONLY the C function code, no explanations."
)

SAFE_PROMPT_TEMPLATE = (
    "Write a SECURE C function that {task}. "
    "The function MUST include: "
    "proper input validation, bounds checking, NULL pointer checks, "
    "error handling for memory allocation failures, and safe buffer operations. "
    "Do NOT use unsafe functions like gets(), strcpy(), strcat(), or sprintf()."
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    human_labels = load_human_labels(CONFIG['human_eval_csv'])

    print("=" * 60)
    print("B4: Safe Prompt Baseline")
    print("=" * 60)

    # 🔑 已按你的要求直接硬编码 DeepSeek 原生接口配置
    API_KEY = "sk-43433e3f388b48498a3d9f5669cd42a2"
    BASE_URL = "https://api.deepseek.com"  # ← 原生接口用根路径，不带 /v1
    MODEL_NAME = "deepseek-v4-pro"          # ← DeepSeek 高级推理模型

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    tokenizer, codebert, classifier = load_models()
    print("Models loaded")

    all_candidates = []
    per_prompt = {}

    for pi, (pid, prompt_text) in enumerate(PROMPTS):
        print(f"\n[{pi+1}/15] {pid}")

        task = prompt_text.replace("Write a C function that ", "").replace("Write a C function to ", "")
        task = task[0].lower() + task[1:] if task else prompt_text
        safe_prompt = SAFE_PROMPT_TEMPLATE.format(task=task)

        candidates = []
        for ci in range(CONFIG['num_candidates']):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_SAFE},
                        {"role": "user", "content": safe_prompt}
                    ],
                    temperature=CONFIG['llm_temperature'],
                    seed=ci * 42,
                    stream=False,
                    reasoning_effort="high",  # ← DeepSeek 特有参数
                    extra_body={"thinking": {"type": "enabled"}}  # ← DeepSeek 特有参数
                )
                code = strip_code(resp.choices[0].message.content)
                p_vul = score_classifier(code, tokenizer, codebert, classifier)
                candidates.append((ci, code, p_vul))
                all_candidates.append({
                    'prompt_id': pid, 'candidate_idx': ci,
                    'code': code, 'p_vul': p_vul, 'p_safe': 1.0 - p_vul,
                })
                print(f"  c{ci}: P(vul)={p_vul:.6f}, {len(code)} chars")
            except Exception as e:
                print(f"  c{ci} FAILED: {e}")

        candidates.sort(key=lambda x: x[2])
        best_ci, best_code, best_pvul = candidates[0]
        per_prompt[pid] = (best_ci, best_pvul)
        human_label = human_labels.get((pid, best_ci), -1)
        label_str = "SAFE" if human_label == 1 else ("UNSAFE" if human_label == 0 else "?")
        print(f"  Best: c{best_ci} P(vul)={best_pvul:.6f} [{label_str}]")

    pd.DataFrame(all_candidates).to_csv(OUTPUT_DIR / 'all_candidates.csv', index=False)

    selections = {pid: ci for pid, (ci, _) in per_prompt.items()}
    safe_picks, total, misses = evaluate_selection(selections, human_labels)
    rate = safe_picks / total * 100 if total > 0 else 0

    print(f"\nB4 Result: {safe_picks}/{total} safe ({rate:.1f}%)")
    if misses:
        print("Misses:")
        for pid, idx, safe_alt in misses:
            print(f"  {pid} c{idx}: {safe_alt} safe alternatives available")

    with open(OUTPUT_DIR / 'result.json', 'w') as f:
        json.dump({
            'baseline': 'B4_SafePrompt',
            'safe_picks': safe_picks, 'total_valid': total, 'safe_rate': rate,
            'avg_p_vul': float(np.mean([pv for _, pv in per_prompt.values()])),
            'selections': {k: int(v) for k, v in selections.items()},
            'misses': [{'prompt_id': p, 'selected_idx': i, 'safe_alternatives': s} for p, i, s in misses],
        }, f, indent=2)

    return safe_picks, total, rate


if __name__ == '__main__':
    main()