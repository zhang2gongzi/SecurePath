#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B6: Reflexion baseline
Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., NeurIPS 2023)
LLM generates → self-reviews security → reflects → regenerates (3 iterations)
"""
import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, PROMPTS, load_human_labels, evaluate_selection
from common import load_models, score_classifier, strip_code

OUTPUT_DIR = Path(CONFIG['output_dir']) / 'b6_reflexion'

REVIEW_PROMPT = """Review this C code for security vulnerabilities. Check for:
- Buffer overflows (CWE-119, CWE-787)
- NULL pointer dereference (CWE-476)
- Use-after-free (CWE-416)
- Integer overflow (CWE-190)
- Memory leaks (CWE-401)
- Missing input validation (CWE-20)
- Unsafe function usage (CWE-676)

For each issue found, state:
1. The exact code location
2. The CWE type
3. A concrete fix

If no issues found, state "NO_ISSUES_FOUND".

CODE TO REVIEW:
```c
{code}
```"""

REGENERATE_PROMPT = """Based on this security review, rewrite the C function to fix ALL identified issues.

ORIGINAL TASK: {task}

SECURITY REVIEW:
{review}

Rewrite the complete C function with all security fixes applied. Output ONLY the corrected code."""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    human_labels = load_human_labels(CONFIG['human_eval_csv'])

    print("=" * 60)
    print("B6: Reflexion Baseline (Self-Reflection + Iterative Repair)")
    print("Reference: Shinn et al., NeurIPS 2023")
    print("=" * 60)

    # 🔑 已按你的要求直接硬编码 DeepSeek 原生接口配置
    API_KEY = "sk-43433e3f388b48498a3d9f5669cd42a2"
    BASE_URL = "https://api.deepseek.com"  # ← 原生接口用根路径，不带 /v1
    MODEL_NAME = "deepseek-v4-pro"          # ← DeepSeek 高级推理模型

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    tokenizer, codebert, classifier = load_models()
    print("Models loaded")

    results = []
    per_prompt = {}
    MAX_ITER = 3

    for pi, (pid, prompt_text) in enumerate(PROMPTS):
        print(f"\n[{pi+1}/15] {pid}")

        # Step 1: Initial generation
        print("  [Iter 1] Generating initial code...")
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a C programmer. Generate ONLY the C function code, no explanations."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=CONFIG['llm_temperature'],
                seed=42,
                stream=False,
                reasoning_effort="high",  # ← DeepSeek 特有参数
                extra_body={"thinking": {"type": "enabled"}}  # ← DeepSeek 特有参数
            )
            code = strip_code(resp.choices[0].message.content)
        except Exception as e:
            print(f"  Initial generation FAILED: {e}")
            continue

        p_vul = score_classifier(code, tokenizer, codebert, classifier)
        print(f"  Initial P(vul)={p_vul:.6f}, {len(code)} chars")

        trajectory = [{'iter': 0, 'p_vul': p_vul, 'code': code, 'review': ''}]

        # Step 2-3: Review → Regenerate loop
        for it in range(1, MAX_ITER):
            # Review
            try:
                review_resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a senior security auditor. Be specific and precise."},
                        {"role": "user", "content": REVIEW_PROMPT.format(code=code)}
                    ],
                    temperature=0.3,
                    stream=False,
                    reasoning_effort="high",
                    extra_body={"thinking": {"type": "enabled"}}
                )
                review = review_resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"  Review FAILED: {e}")
                break

            no_issues = "NO_ISSUES_FOUND" in review
            print(f"  [Iter {it+1}] Review: {'NO ISSUES' if no_issues else 'issues found'} ({len(review)} chars)")

            if no_issues:
                trajectory.append({'iter': it, 'p_vul': p_vul, 'code': code, 'review': review})
                break

            # Regenerate based on review
            try:
                regen_resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a C programmer fixing security issues. Output ONLY the corrected C function code."},
                        {"role": "user", "content": REGENERATE_PROMPT.format(task=prompt_text, review=review)}
                    ],
                    temperature=CONFIG['llm_temperature'],
                    seed=42 + it,
                    stream=False,
                    reasoning_effort="high",
                    extra_body={"thinking": {"type": "enabled"}}
                )
                code = strip_code(regen_resp.choices[0].message.content)
                p_vul = score_classifier(code, tokenizer, codebert, classifier)
                print(f"  [Iter {it+1}] Regenerated P(vul)={p_vul:.6f}, {len(code)} chars")
                trajectory.append({'iter': it, 'p_vul': p_vul, 'code': code, 'review': review})
            except Exception as e:
                print(f"  Regenerate FAILED: {e}")
                break

        # Track P(vul) trajectory
        pvuls = [t['p_vul'] for t in trajectory]
        best_iter = int(np.argmin(pvuls))
        best_code = trajectory[best_iter]['code']
        best_pvul = pvuls[best_iter]

        per_prompt[pid] = (0, best_pvul)
        results.append({
            'prompt_id': pid,
            'final_code': best_code,
            'best_p_vul': best_pvul,
            'best_iter': best_iter,
            'p_vul_trajectory': pvuls,
            'n_iters': len(trajectory),
        })

        human_label = human_labels.get((pid, 0), -1)
        label_str = "SAFE" if human_label == 1 else ("UNSAFE" if human_label == 0 else "?")
        traj_str = " → ".join(f"R{ti['iter']}:{ti['p_vul']:.4f}" for ti in trajectory)
        print(f"  Trajectory: {traj_str}")
        print(f"  Best: iter={best_iter} P(vul)={best_pvul:.6f} [human_label_c0: {label_str}]")

    # Evaluate (using candidate_idx=0 as proxy; new human eval needed for final scores)
    selections = {pid: 0 for pid in per_prompt}
    safe_picks, total, misses = evaluate_selection(selections, human_labels)
    rate = safe_picks / total * 100 if total > 0 else 0

    avg_pvul = float(np.mean([pv for _, pv in per_prompt.values()]))
    improved = sum(1 for r in results if len(r['p_vul_trajectory']) >= 2 and r['p_vul_trajectory'][-1] < r['p_vul_trajectory'][0])
    degraded = sum(1 for r in results if len(r['p_vul_trajectory']) >= 2 and r['p_vul_trajectory'][-1] > r['p_vul_trajectory'][0])

    print(f"\nB6 Result: {safe_picks}/{total} safe ({rate:.1f}%) [approximate, needs new human eval]")
    print(f"  Avg P(vul): {avg_pvul:.6f}")
    print(f"  Improved: {improved}/15, Degraded: {degraded}/15")
    print(f"  NOTE: human labels are for original experiment_b code (candidate_idx=0 only).")
    print(f"  New human evaluation required for Reflexion-generated code.")

    # Save trajectory data for analysis
    rows = []
    for r in results:
        for t_idx, pv in enumerate(r['p_vul_trajectory']):
            rows.append({'prompt_id': r['prompt_id'], 'iteration': t_idx, 'p_vul': pv})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'trajectories.csv', index=False)

    with open(OUTPUT_DIR / 'result.json', 'w') as f:
        json.dump({
            'baseline': 'B6_Reflexion',
            'reference': 'Shinn et al., NeurIPS 2023',
            'safe_picks': safe_picks, 'total_valid': total, 'safe_rate': rate,
            'avg_p_vul': avg_pvul,
            'n_improved': improved, 'n_degraded': degraded,
            'note': 'Human labels are approximate (candidate_idx=0 proxy). New human eval needed.',
            'selections': {k: int(v) for k, v in selections.items()},
            'per_prompt': [{
                'prompt_id': r['prompt_id'],
                'best_p_vul': r['best_p_vul'],
                'best_iter': r['best_iter'],
                'p_vul_trajectory': r['p_vul_trajectory'],
                'n_iters': r['n_iters'],
            } for r in results],
        }, f, indent=2)

    print(f"\nOutput: {OUTPUT_DIR}")
    return safe_picks, total, rate


if __name__ == '__main__':
    main()