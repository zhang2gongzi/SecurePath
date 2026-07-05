#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ISR-random 消融：随机 token attention 替代真实分类器 attention
用于验证空间定位信号的因果有效性

对照 ISR-2：除了 attention 来源外，其他完全相同
  - ISR-2:   真实分类器 attention → 定位高风险 token
  - ISR-random: 随机选择代码中的 token → 伪定位

如果 ISR-random 显著差于 ISR-2，直接证明 attention 空间信号有效。
"""
import os, sys, json, re, torch, numpy as np, random, pandas as pd
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from anthropic import Anthropic

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from attention_pooling import SafetyClassifier

CONFIG = {
    'anthropic_api_key': os.getenv('ANTHROPIC_AUTH_TOKEN', 'eyJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJPMi1ITVMtU0VSVklDRSIsInN1YiI6InpoYW5nemwxIiwiYXVkIjoiUG9ydGFsLWFpLW1jcCIsIndvcmtObyI6IkcyNjAyMDIwIiwiZXhwIjoxNzgzMDA3OTk5LCJuYmYiOjE3ODAzMjk2MDAsImlhdCI6MTc4MDM3MTY2NCwianRpIjoiNjdlNDQ3MGMtMmE2MS00MmNhLTg3YWEtZDA4N2U5NzY1YWFjIn0.NYDK0uDKW4Ol0AvpFUd9CkjVfANq1CSDGy5V4ZwGIbrdQNRUkHn1zrPZfilRRs8y3bGLd13BDzu_GdMpd2-i3S0KKMY6PIE_5Te8Ht2d3rS_kdJnnnDsqTfksESldnaeN68yAYMxDE53gfYSD2_rBOWnL-zPPjkoiuLLsaF2a5RfoHLD9GAU5LLjyXJKPQ7fHQ2my3NU1XVToqtbJ9UuOfwYAWcew-cwUo4KLnShS7mm7rYAhB_Z04QCMDvlFTirnLVxH7hF0D19HusaXo09rnFeOnSTWwQFmUCrPXbPzYWAF1yYQ2Ke57AJHVIAdYj9pKDdwPomRH4jqAwr6YRQCw'),
    'anthropic_base_url': os.getenv('ANTHROPIC_BASE_URL', 'https://ai-gateway-inside.haplat.net/devops'),
    'llm_model': os.getenv('LLM_MODEL', 'wangsu-anthropic-glm-latest'),
    'llm_temperature': 1.2,
    'feedback_temperature': 0.8,
    'max_tokens': 4096,
    'max_iterations': 5,
    'stagnation_patience': 3,

    'encoder_path': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/models/best_model',
    'classifier_path': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/models/safety_classifier_attn.pt',
    'output_dir': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/experiment_isr_random',

    'max_length': 512,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
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

ATTENTION_FEEDBACK = (
    "\n\n## SECURITY REVIEW - ATTENTION HEATMAP\n"
    "A safety classifier (trained on real-world CVE data) analyzed the code above "
    "and gave it a risk score of P(vul) = {p_vul:.4f}.\n\n"
    "The classifier's attention mechanism flagged these specific code regions "
    "as most indicative of vulnerability:\n\n"
    "{flagged_regions}\n\n"
    "## REQUIRED FIXES\n"
    "{required_fixes}\n\n"
    "Please rewrite the COMPLETE function addressing ALL flagged issues. "
    "Generate ONLY the corrected C function code:"
)

# C 代码中可被标记的 token 模式
CODE_SPAN_PATTERNS = [
    (r'\b(strcpy|strcat|sprintf|gets|scanf|printf)\s*\([^;]*\)', 'function call'),
    (r'\b(malloc|calloc|realloc|alloca)\s*\([^;]*\)', 'allocation'),
    (r'\b(free|delete)\s*\([^;]*\)', 'deallocation'),
    (r'\b(memcpy|memmove|memset|memcmp)\s*\([^;]*\)', 'memory operation'),
    (r'\b(fread|fwrite|fopen|fclose|fgets|fputs)\s*\([^;]*\)', 'file I/O'),
    (r'\b(strncpy|strncat|snprintf|strlcpy|strlcat)\s*\([^;]*\)', 'bounded string op'),
    (r'\b(while|for)\s*\([^)]*\)', 'loop'),
    (r'\[\s*\w+\s*\]', 'array access'),
    (r'->\s*\w+', 'pointer deref'),
    (r'\b(strlen|strcmp|strncmp|strdup)\s*\([^;]*\)', 'string operation'),
    (r'\b(return|goto)\b', 'control flow'),
]


def extract_random_spans(code, n_spans=3):
    """从代码中随机选择 n_spans 个代码片段，模拟 attention 输出"""
    # 收集所有可匹配的代码 span
    all_spans = []
    for pattern, category in CODE_SPAN_PATTERNS:
        for m in re.finditer(pattern, code, re.IGNORECASE):
            span_text = m.group(0).strip()
            if len(span_text) >= 3:  # 过滤太短的
                all_spans.append((span_text[:80], category))

    if len(all_spans) < n_spans:
        # 如果匹配不够，从行级文本中补
        lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith('//')]
        for line in lines:
            if len(line) > 10:
                all_spans.append((line[:80], 'code line'))

    if not all_spans:
        return "  (No specific regions identified)", []

    random.shuffle(all_spans)
    selected = all_spans[:n_spans]

    flagged_lines = []
    risky_patterns = []
    for rank, (text, cat) in enumerate(selected, 1):
        bar = '█' * random.randint(3, 8)  # 随机注意力条
        attn_val = random.uniform(0.0005, 0.0015)  # 随机注意力值
        flagged_lines.append(
            f'  [{rank}] "{text}"\n'
            f'      [random attention] {bar} ({attn_val:.4f})'
        )
        risky_patterns.append(text)

    return '\n'.join(flagged_lines), risky_patterns


def generate_required_fixes(risky_patterns, prompt_text):
    """基于（随机选出的）模式生成修复建议 — 保持外观与真实 ISR-2 一致"""
    fixes = []
    patterns_str = ' '.join(risky_patterns).lower()
    task_lower = prompt_text.lower()

    if any(kw in patterns_str for kw in ['strcpy', 'strcat', 'sprintf', 'gets', 'scanf']):
        fixes.append("- Replace unsafe string functions with bounded alternatives (strncpy, snprintf)")
    if any(kw in patterns_str for kw in ['while', 'for', '++', 'dst', 'dest', 'copy']) and 'bounds' in task_lower:
        fixes.append("- Add explicit bounds checking before copying data")
    if any(kw in patterns_str for kw in ['malloc', 'alloc', 'calloc', 'realloc']):
        if not any(kw in patterns_str for kw in ['null', '!ptr', 'if (!', '== null']):
            fixes.append("- Add NULL check after every memory allocation")
    if any(kw in patterns_str for kw in ['free']):
        fixes.append("- Set pointer to NULL after free to prevent use-after-free")
    if any(kw in patterns_str for kw in ['index', 'array', '[']):
        fixes.append("- Validate array index is within bounds before access")
    if any(kw in patterns_str for kw in ['realloc']):
        fixes.append("- Save realloc() result to a temp variable to preserve original on failure")
    if any(kw in patterns_str for kw in ['memcpy', 'memmove']):
        fixes.append("- Use memmove() instead of memcpy() for potentially overlapping regions")
    if any(kw in patterns_str for kw in ['strlen', 'str', 'string']) and 'concat' in task_lower:
        fixes.append("- Validate input string pointers are non-NULL before strlen()")

    if not fixes:
        fixes = [
            "- Validate all input parameters before use",
            "- Add bounds checking for buffer operations",
            "- Check return values of all allocation functions",
        ]

    return '\n'.join(fixes[:5])


def load_models():
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['encoder_path'])
    codebert = AutoModel.from_pretrained(CONFIG['encoder_path']).to(CONFIG['device'])
    codebert.eval()
    classifier = SafetyClassifier().to(CONFIG['device'])
    classifier.load_state_dict(torch.load(CONFIG['classifier_path'], map_location=CONFIG['device']))
    classifier.eval()
    return tokenizer, codebert, classifier


@torch.no_grad()
def analyze_code(code, tokenizer, codebert, classifier, max_length=512):
    device = CONFIG['device']
    enc = tokenizer(code, max_length=max_length, padding='max_length',
                    truncation=True, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    outputs = codebert(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = outputs.last_hidden_state

    scores = classifier.attention_pool.query(hidden_states).squeeze(-1)
    scores = scores.masked_fill(attention_mask == 0, -1e9)
    weights = torch.nn.functional.softmax(scores, dim=-1)

    logits = classifier(hidden_states, attention_mask)
    p_vul = torch.nn.functional.softmax(logits, dim=1)[0, 1].item()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    attn_weights = weights[0].cpu().tolist()

    return p_vul, tokens, attn_weights


def generate_initial(prompt_text, client, model):
    resp = client.messages.create(
        model=model,
        max_tokens=CONFIG['max_tokens'],
        system=SYSTEM_CODER,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=CONFIG['llm_temperature'],
    )
    code = resp.content[0].text.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return code


def fix_with_feedback(feedback_prompt, client, model):
    resp = client.messages.create(
        model=model,
        max_tokens=CONFIG['max_tokens'],
        system=SYSTEM_FIXER,
        messages=[{"role": "user", "content": feedback_prompt}],
        temperature=CONFIG['feedback_temperature'],
    )
    code = resp.content[0].text.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return code


def run_isr_random(prompt_id, prompt_text, client, models, seed=None):
    """ISR-random: 使用随机 token 定位替代真实 attention"""
    tokenizer, codebert, classifier = models
    if seed is not None:
        random.seed(seed + hash(prompt_id) % 100000)
        np.random.seed(seed + hash(prompt_id) % 100000)

    max_iters = CONFIG['max_iterations']
    full_prompt = prompt_text

    iterations = []
    best_code = None
    best_p_vul = float('inf')
    stagnation_count = 0

    for iteration in range(max_iters):
        if iteration == 0:
            code = generate_initial(full_prompt, client, CONFIG['llm_model'])
        else:
            prev_code = iterations[-1]['code']
            prev_p_vul = iterations[-1]['p_vul']

            # 使用随机 token 定位替代真实 attention
            flagged_text, risky_patterns = extract_random_spans(prev_code, n_spans=random.randint(2, 4))
            required_fixes = generate_required_fixes(risky_patterns, prompt_text)

            feedback = prev_code + ATTENTION_FEEDBACK.format(
                p_vul=prev_p_vul,
                flagged_regions=flagged_text,
                required_fixes=required_fixes,
            )
            code = fix_with_feedback(feedback, client, CONFIG['llm_model'])

        if not code.strip():
            break

        p_vul, tokens, attn_weights = analyze_code(code, tokenizer, codebert, classifier)

        iter_record = {
            'iteration': iteration,
            'code': code,
            'p_vul': p_vul,
            'code_len': len(code),
        }
        iterations.append(iter_record)

        if p_vul < best_p_vul:
            best_p_vul = p_vul
            stagnation_count = 0
        else:
            stagnation_count += 1

        status = "✓" if p_vul < 0.001 else "→"
        print(f"    [ISR-random] Iter {iteration}: P(vul)={p_vul:.6f} {status}")

        if stagnation_count >= CONFIG['stagnation_patience'] and iteration > 1:
            print(f"    [ISR-random] Stagnated after {iteration+1} iterations")
            break

    best_idx = min(range(len(iterations)), key=lambda i: iterations[i]['p_vul'])
    return {
        'prompt_id': prompt_id,
        'ablation_id': 'ISR-random',
        'iterations': iterations,
        'num_iterations': len(iterations),
        'best_iteration': best_idx,
        'best_p_vul': iterations[best_idx]['p_vul'],
        'initial_p_vul': iterations[0]['p_vul'] if iterations else None,
        'p_vul_reduction': (iterations[0]['p_vul'] - iterations[best_idx]['p_vul']) if iterations else 0,
    }


def main():
    print("=" * 60)
    print("ISR-random 消融实验")
    print("随机 token attention 替代真实分类器 attention")
    print("=" * 60)

    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading models...")
    models = load_models()
    print("Models loaded")

    client = Anthropic(
        api_key=CONFIG['anthropic_api_key'],
        base_url=CONFIG['anthropic_base_url'],
    )

    seed = CONFIG.get('random_seed', 42)
    per_prompt = []

    for pi, (prompt_id, prompt_text) in enumerate(PROMPTS):
        print(f"\n[{pi+1}/{len(PROMPTS)}] {prompt_id}")
        result = run_isr_random(prompt_id, prompt_text, client, models, seed=seed)
        per_prompt.append(result)

        init_p = result['initial_p_vul']
        best_p = result['best_p_vul']
        red = result['p_vul_reduction']
        print(f"  Init={init_p:.6f} → Best={best_p:.6f} (Δ={red:+.6f}), "
              f"{result['num_iterations']} iters")

    # 保存迭代级数据
    summary_rows = []
    for r in per_prompt:
        for it in r['iterations']:
            summary_rows.append({
                'ablation_id': 'ISR-random',
                'prompt_id': r['prompt_id'],
                'iteration': it['iteration'],
                'p_vul': it['p_vul'],
                'code_len': it['code_len'],
                'code': it['code'],
            })
    pd.DataFrame(summary_rows).to_csv(output_dir / 'iterations_ISR_random.csv', index=False)

    # 汇总
    print(f"\n{'='*60}")
    print("SUMMARY: ISR-random")
    print(f"{'='*60}")
    init_avg = np.mean([r['initial_p_vul'] for r in per_prompt if r['initial_p_vul'] is not None])
    best_avg = np.mean([r['best_p_vul'] for r in per_prompt])
    red_avg = np.mean([r['p_vul_reduction'] for r in per_prompt])
    iter_avg = np.mean([r['num_iterations'] for r in per_prompt])
    print(f"  Init P(vul):  {init_avg:.6f}")
    print(f"  Best P(vul):  {best_avg:.6f}")
    print(f"  Δ P(vul):     {red_avg:+.6f}")
    print(f"  Avg Iters:    {iter_avg:.1f}")

    report = {
        'config': {k: str(v) for k, v in CONFIG.items() if 'key' not in k.lower()},
        'ablation_id': 'ISR-random',
        'num_prompts': len(per_prompt),
        'avg_initial_p_vul': float(init_avg),
        'avg_best_p_vul': float(best_avg),
        'avg_p_vul_reduction': float(red_avg),
        'avg_iterations': float(iter_avg),
    }
    with open(output_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved: {output_dir}")


if __name__ == '__main__':
    main()
