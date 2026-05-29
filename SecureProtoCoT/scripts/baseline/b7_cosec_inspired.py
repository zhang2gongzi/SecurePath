#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B7: CoSec-inspired baseline
CoSec (Li et al., ISSTA 2024): security model co-decodes per-token for safe generation.
Original requires white-box model access. This baseline approximates the paradigm via
prompt engineering: a "security checker" prompt that validates code in multiple stages.

Two-stage generation:
  Stage 1: LLM generates security spec for the task
  Stage 2: LLM writes code line-by-line, self-checking security after each logical block
"""
import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, PROMPTS, load_human_labels, evaluate_selection
from common import load_models, score_classifier, strip_code

OUTPUT_DIR = Path(CONFIG['output_dir']) / 'b7_cosec_inspired'

SYSTEM_COSEC = (
    "You are a security-hardened code generator. Your code generation process is "
    "monitored by a security checker that validates each statement against CWE "
    "vulnerability patterns. Before writing any code, you MUST:\n"
    "1. List the security properties this function must satisfy\n"
    "2. Write the code with inline security annotations (/* SEC: ... */) after each "
    "   critical statement, explaining why it is safe\n"
    "3. After the function, add a SECURITY SELF-AUDIT section listing each check performed\n\n"
    "This mimics the CoSec (ISSTA 2024) co-decoding paradigm where a security model "
    "reviews each token during generation."
)

COSEC_PROMPT = """Write a secure C function that {task}.

Follow this exact format:

## SECURITY REQUIREMENTS
[List 3-5 specific security properties this function must guarantee]

## CODE
[The C function with /* SEC: ... */ annotations after critical lines]

## SELF-AUDIT
[Check each requirement against the written code. State PASS or FAIL with reason.]"""


def parse_code_from_response(text):
    """Extract code from CoSec-formatted response (between ## CODE and ## SELF-AUDIT)."""
    if "## CODE" in text and "## SELF-AUDIT" in text:
        code_block = text.split("## CODE")[1].split("## SELF-AUDIT")[0]
    elif "## CODE" in text:
        code_block = text.split("## CODE")[1]
    else:
        code_block = text

    code_block = code_block.strip()
    if "```" in code_block:
        lines = code_block.split("\n")
        start = next((i for i, l in enumerate(lines) if "```" in l), -1)
        end = next((i for i, l in enumerate(lines) if i > start and "```" in l), -1)
        if start >= 0 and end >= 0:
            code_block = "\n".join(lines[start+1:end])

    return code_block.strip()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    human_labels = load_human_labels(CONFIG['human_eval_csv'])

    print("=" * 60)
    print("B7: CoSec-inspired Baseline (Security Co-Decoding via Prompt)")
    print("Reference: Li et al., ISSTA 2024 - prompt-based approximation")
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

        candidates = []
        for ci in range(CONFIG['num_candidates']):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_COSEC},
                        {"role": "user", "content": COSEC_PROMPT.format(task=task)}
                    ],
                    temperature=CONFIG['llm_temperature'],
                    seed=ci * 42,
                    stream=False,
                    reasoning_effort="high",  # ← DeepSeek 特有参数
                    extra_body={"thinking": {"type": "enabled"}}  # ← DeepSeek 特有参数
                )
                full_response = resp.choices[0].message.content
                code = parse_code_from_response(full_response)
                p_vul = score_classifier(code, tokenizer, codebert, classifier)
                candidates.append((ci, code, p_vul, full_response))
                all_candidates.append({
                    'prompt_id': pid, 'candidate_idx': ci,
                    'code': code, 'p_vul': p_vul, 'p_safe': 1.0 - p_vul,
                    'has_security_section': '## SECURITY REQUIREMENTS' in full_response,
                    'has_self_audit': '## SELF-AUDIT' in full_response,
                })
                print(f"  c{ci}: P(vul)={p_vul:.6f}, {len(code)} chars")

                # Check audit results
                if '## SELF-AUDIT' in full_response:
                    audit = full_response.split('## SELF-AUDIT')[1]
                    n_pass = audit.count('PASS')
                    n_fail = audit.count('FAIL')
                    print(f"        Self-audit: {n_pass}P/{n_fail}F")
            except Exception as e:
                print(f"  c{ci} FAILED: {e}")

        candidates.sort(key=lambda x: x[2])
        best_ci, best_code, best_pvul, _ = candidates[0]
        per_prompt[pid] = (best_ci, best_pvul)
        human_label = human_labels.get((pid, best_ci), -1)
        label_str = "SAFE" if human_label == 1 else ("UNSAFE" if human_label == 0 else "?")
        print(f"  Best: c{best_ci} P(vul)={best_pvul:.6f} [{label_str}]")

    pd.DataFrame(all_candidates).to_csv(OUTPUT_DIR / 'all_candidates.csv', index=False)

    selections = {pid: ci for pid, (ci, _) in per_prompt.items()}
    safe_picks, total, misses = evaluate_selection(selections, human_labels)
    rate = safe_picks / total * 100 if total > 0 else 0

    avg_pvul = float(np.mean([pv for _, pv in per_prompt.values()]))
    compliant = sum(1 for c in all_candidates if c['has_security_section'] and c['has_self_audit'])

    print(f"\nB7 Result: {safe_picks}/{total} safe ({rate:.1f}%)")
    print(f"  Avg P(vul): {avg_pvul:.6f}")
    print(f"  CoSec compliance (has both sections): {compliant}/{len(all_candidates)}")
    if misses:
        print("Misses:")
        for pid, idx, safe_alt in misses:
            print(f"  {pid} c{idx}: {safe_alt} safe alternatives available")

    with open(OUTPUT_DIR / 'result.json', 'w') as f:
        json.dump({
            'baseline': 'B7_CoSec_Inspired',
            'reference': 'Li et al., ISSTA 2024 - prompt-based approximation',
            'safe_picks': safe_picks, 'total_valid': total, 'safe_rate': rate,
            'avg_p_vul': avg_pvul,
            'n_cosec_compliant': compliant,
            'note': 'Prompt-based simulation of co-decoding. True CoSec requires white-box token-level intervention.',
            'selections': {k: int(v) for k, v in selections.items()},
            'misses': [{'prompt_id': p, 'selected_idx': i, 'safe_alternatives': s} for p, i, s in misses],
        }, f, indent=2)

    print(f"\nOutput: {OUTPUT_DIR}")
    return safe_picks, total, rate


if __name__ == '__main__':
    main()