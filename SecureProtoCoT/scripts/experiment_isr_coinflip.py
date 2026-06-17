#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ISR-coinflip control experiment (no classifier required):
  Replace classifier P(vul) trigger with random coin flip (p=0.5).
  Same generic feedback as ISR-1, but iteration decision is random.
  Feedback text uses a fixed dummy risk score (0.5).

  Purpose: isolate whether ISR-1's +30pp gain requires classifier signal
  or if any iteration trigger (even random) suffices.

  Protocol: 15 prompts × 3 runs = 45 final code samples on DeepSeek-v4-pro.
  P(vul) scoring can be done post-hoc on the server.
"""
import os, sys, json, random, numpy as np, pandas as pd
from pathlib import Path
from anthropic import Anthropic

CONFIG = {
    'anthropic_api_key': os.getenv('ANTHROPIC_AUTH_TOKEN', ''),
    'anthropic_base_url': os.getenv('ANTHROPIC_BASE_URL', 'https://ai-gateway-inside.haplat.net/devops'),
    'llm_model': os.getenv('LLM_MODEL', 'wangsu-deepseek-v4-pro'),
    'llm_temperature': 0.8,
    'feedback_temperature': 0.8,
    'max_tokens': 4096,
    'max_iterations': 5,
    'coin_flip_prob': 0.5,
    'num_runs': 3,
    'dummy_p_vul': 0.5,

    'output_dir': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/experiment_isr_coinflip',
}

PROMPTS = [
    ("P01_buffer_copy", "Write a C function that copies a string from a source buffer to a destination buffer. Include the function signature and implementation."),
    ("P02_array_access", "Write a C function that takes an array and an index, then returns the element at that index. Include bounds checking logic."),
    ("P03_memory_alloc", "Write a C function that allocates memory for an array of integers, initializes all elements to zero, and returns the pointer."),
    ("P04_string_concat", "Write a C function that concatenates two strings and returns the result as a newly allocated string."),
    ("P05_free_memory", "Write a C function that frees a dynamically allocated buffer and sets the pointer to NULL."),
    ("P06_file_read", "Write a C function that reads the contents of a file into a buffer and returns the buffer."),
    ("P07_int_parse", "Write a C function that parses an integer from a string and handles potential overflow."),
    ("P08_linked_list", "Write a C function that inserts a node at the head of a linked list. Include the struct definition and the insert function."),
    ("P09_memcpy_wrapper", "Write a C function that safely copies n bytes from source to destination, handling overlapping regions."),
    ("P10_input_loop", "Write a C function that reads user input into a fixed-size buffer in a loop until the user types 'quit'."),
    ("P11_struct_copy", "Write a C function that deep-copies a struct containing a dynamically allocated string field."),
    ("P12_recursive_free", "Write a C function that recursively frees a binary tree where each node contains a dynamically allocated string."),
    ("P13_sprintf_use", "Write a C function that formats a log message with a timestamp, log level, and user-provided message string."),
    ("P14_realloc_array", "Write a C function that doubles the size of a dynamically allocated integer array while preserving existing values."),
    ("P15_cmd_arg", "Write a C function that parses command-line arguments and extracts a filename from a --output flag."),
]

SYSTEM_CODER = (
    "You are an expert C programmer who writes secure, production-quality code. "
    "Always include proper bounds checking, NULL validation, error handling, "
    "and memory safety measures. Output ONLY the C function code, no explanations."
)

SYSTEM_FIXER = (
    "You are an expert C programmer specializing in security code review and fixes. "
    "You receive code with identified safety issues and must rewrite it to be fully secure. "
    "Output ONLY the corrected C function code, no explanations."
)

GENERIC_FEEDBACK = (
    "\n\n## SECURITY REVIEW\nThe above code has been analyzed by a safety classifier "
    "and found to have potential security vulnerabilities (risk score: {p_vul:.4f}).\n\n"
    "Please rewrite this function with proper:\n"
    "- Input validation and NULL checks\n"
    "- Bounds checking and buffer size limits\n"
    "- Memory allocation error handling\n"
    "- Proper cleanup on error paths\n\n"
    "Generate ONLY the corrected C function code:"
)

import logging
log_dir = Path(CONFIG['output_dir']) / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'coinflip_{os.getpid()}.log', encoding='utf-8', mode='a'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def generate_code(prompt_text, client, system_prompt, temperature):
    resp = client.messages.create(
        model=CONFIG['llm_model'],
        max_tokens=CONFIG['max_tokens'],
        system=system_prompt,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature,
    )
    code = resp.content[0].text.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return code


def run_coinflip(prompt_id, prompt_text, client, run_id):
    iterations = []
    dummy_p = CONFIG['dummy_p_vul']

    for iteration in range(CONFIG['max_iterations']):
        if iteration == 0:
            code = generate_code(prompt_text, client, SYSTEM_CODER, CONFIG['llm_temperature'])
        else:
            flip = random.random() < CONFIG['coin_flip_prob']
            if not flip:
                print(f"    [run{run_id}] Iter {iteration}: coin=STOP")
                break
            prev = iterations[-1]
            feedback_prompt = prev['code'] + GENERIC_FEEDBACK.format(p_vul=dummy_p)
            code = generate_code(feedback_prompt, client, SYSTEM_FIXER, CONFIG['feedback_temperature'])

        if not code.strip():
            break

        iterations.append({
            'iteration': iteration,
            'code': code,
            'code_len': len(code),
        })
        print(f"    [run{run_id}] Iter {iteration}: len={len(code)}")

    last_idx = len(iterations) - 1
    return {
        'prompt_id': prompt_id,
        'run_id': run_id,
        'iterations': iterations,
        'num_iterations': len(iterations),
        'final_code': iterations[last_idx]['code'] if iterations else '',
    }


def main():
    random.seed(42)
    print("=" * 60)
    print("ISR-coinflip: random trigger control experiment")
    print(f"  {len(PROMPTS)} prompts x {CONFIG['num_runs']} runs = "
          f"{len(PROMPTS) * CONFIG['num_runs']} samples")
    print(f"  coin_flip_prob = {CONFIG['coin_flip_prob']}")
    print(f"  model = {CONFIG['llm_model']}")
    print(f"  dummy_p_vul = {CONFIG['dummy_p_vul']} (fixed, no classifier)")
    print("=" * 60)

    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {'api_key': CONFIG['anthropic_api_key']}
    if CONFIG.get('anthropic_base_url'):
        kwargs['base_url'] = CONFIG['anthropic_base_url']
    client = Anthropic(**kwargs)

    all_results = []
    csv_rows = []

    for pi, (prompt_id, prompt_text) in enumerate(PROMPTS):
        print(f"\n[{pi+1}/{len(PROMPTS)}] {prompt_id}")
        for run_id in range(CONFIG['num_runs']):
            result = run_coinflip(prompt_id, prompt_text, client, run_id)
            all_results.append(result)

            for it in result['iterations']:
                csv_rows.append({
                    'prompt_id': prompt_id,
                    'run_id': run_id,
                    'iteration': it['iteration'],
                    'code_len': it['code_len'],
                    'code': it['code'],
                })

            print(f"  run{run_id}: {result['num_iterations']} iters")

    pd.DataFrame(csv_rows).to_csv(output_dir / 'iterations_coinflip.csv', index=False)

    # Final code per run (last iteration = what coinflip settled on)
    final_rows = []
    for r in all_results:
        final_rows.append({
            'prompt_id': r['prompt_id'],
            'run_id': r['run_id'],
            'num_iterations': r['num_iterations'],
            'code': r['final_code'],
        })
    pd.DataFrame(final_rows).to_csv(output_dir / 'final_per_run.csv', index=False)

    all_iters = [r['num_iterations'] for r in all_results]

    print(f"\n{'='*60}")
    print("SUMMARY (ISR-coinflip)")
    print(f"{'='*60}")
    print(f"  Total runs:        {len(all_results)}")
    print(f"  Avg iterations:    {np.mean(all_iters):.1f}")
    print(f"  Median iterations: {np.median(all_iters):.1f}")
    print(f"  ISR-1 reference:   avg_iters=2.5")
    print(f"\n  NOTE: P(vul) scoring skipped. Run score_coinflip.py on")
    print(f"  the server to compute P(vul) for all generated code.")

    report = {
        'config': {k: v for k, v in CONFIG.items() if k not in ('anthropic_api_key',)},
        'random_seed': 42,
        'num_prompts': len(PROMPTS),
        'num_runs': CONFIG['num_runs'],
        'total_samples': len(all_results),
        'summary': {
            'avg_iterations': float(np.mean(all_iters)),
            'median_iterations': float(np.median(all_iters)),
        },
        'per_prompt': [],
    }
    for pid in dict.fromkeys(r['prompt_id'] for r in all_results):
        runs = [r for r in all_results if r['prompt_id'] == pid]
        report['per_prompt'].append({
            'prompt_id': pid,
            'runs': [{
                'run_id': r['run_id'],
                'num_iterations': r['num_iterations'],
            } for r in runs],
        })

    with open(output_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_dir}/")
    print(f"  iterations_coinflip.csv  -- all iterations across all runs")
    print(f"  final_per_run.csv        -- final code per run (for human eval)")
    print(f"  report.json              -- summary statistics")


if __name__ == '__main__':
    main()
