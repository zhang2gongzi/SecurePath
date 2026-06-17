"""
Analyze root causes of BigVul→LLM domain shift.
Compares code features between BigVul training data and LLM-generated code
to explain why the classifier (trained on BigVul) fails on LLM outputs.

Features analyzed:
1. Code length (tokens)
2. Comment ratio
3. Dangerous function usage
4. Structural patterns
"""

import pandas as pd
import numpy as np
import re
import os
import pickle

# ---- Config ----
BASE = r'E:\paper\new\SecureProtoCoT\outputs'
BIGVUL_PATH = r'E:\paper\new\database\MSR_data_cleaned\MSR_data_cleaned.csv'
OUTPUT = os.path.join(BASE, 'domain_shift_root_cause.txt')
BIGVUL_SAMPLE_N = 50000  # sample from BigVul for speed

# Dangerous C functions (memory safety)
DANGEROUS_FUNCS = [
    'strcpy', 'strcat', 'sprintf', 'gets', 'scanf', 'getwd',
    'strlen',  # not dangerous per se but often misused
    'memcpy', 'strncpy', 'strncat', 'snprintf',  # safer but can be misused
    'malloc', 'calloc', 'realloc', 'free',
    'alloca', 'strdup', 'strndup',
    'memmove', 'memset',
    'fopen', 'fread', 'fwrite', 'fgets', 'fclose',
    'read', 'write', 'open', 'close',
    'system', 'exec', 'popen',
]

def extract_c_function(code_str):
    """Extract the C function body from a code string."""
    if not isinstance(code_str, str) or not code_str.strip():
        return ""
    # Remove markdown code blocks
    code_str = re.sub(r'```\w*\n?', '', code_str)
    return code_str.strip()

def count_tokens(code):
    """Approximate token count using whitespace splitting."""
    if not code:
        return 0
    # Simple tokenization: split on whitespace + punctuation
    tokens = re.findall(r'\b\w+\b|[{}();\[\]<>&*+\-/%!=^|~,.]|\"(?:\\.|[^\"\\])*\"|\'(?:\\.|[^\'\\])*\'', code)
    return len(tokens)

def comment_ratio(code):
    """Calculate ratio of comment lines to total lines."""
    if not code:
        return 0
    lines = code.split('\n')
    if len(lines) == 0:
        return 0
    comment_lines = sum(1 for l in lines if l.strip().startswith('//') or
                       l.strip().startswith('/*') or l.strip().startswith('*') or
                       l.strip().startswith('#'))
    return comment_lines / len(lines)

def count_dangerous_calls(code):
    """Count occurrences of dangerous/safety-relevant C functions."""
    if not code:
        return {}
    counts = {}
    for func in DANGEROUS_FUNCS:
        # Match function calls: func( but not func_something(
        pattern = r'\b' + re.escape(func) + r'\s*\('
        matches = re.findall(pattern, code)
        if matches:
            counts[func] = len(matches)
    return counts

def code_stats(code):
    """Compute all stats for a code string."""
    code = extract_c_function(code)
    tokens = count_tokens(code)
    cmt = comment_ratio(code)
    dc = count_dangerous_calls(code)
    n_lines = len(code.split('\n')) if code else 0
    return {
        'tokens': tokens,
        'lines': n_lines,
        'comment_ratio': cmt,
        'dangerous_calls': dc,
        'total_dangerous': sum(dc.values()),
        'unique_dangerous': len(dc),
    }


def load_bigvul_sample():
    """Load BigVul sample from pre-computed pickle file."""
    pkl_path = os.path.join(BASE, 'bigvul_sampled.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        vul = pd.Series(data['vul']).dropna()
        safe = pd.Series(data['safe']).dropna()
        print(f"Loaded from {pkl_path}: {len(vul)} vul, {len(safe)} safe")
        return vul, safe, safe  # safe_changed same as safe for pickle
    raise FileNotFoundError(f"Pickle not found: {pkl_path}. Run sample_bigvul.py first.")


def load_llm_code():
    """Load all LLM-generated code from human evaluation CSVs."""
    codes = []
    labels = []  # human_label: 1=SAFE, 0=UNSAFE
    models = []

    files = {
        'DeepSeek-v4-pro': (os.path.join(BASE, 'human_eval_blinded.csv'), '\t', 'gbk'),
        'GLM-5.1': (os.path.join(BASE, 'claude_experiment', 'human_eval_claude_80.csv'), ',', 'utf-8'),
    }

    for fname, (path, sep, encoding) in files.items():
        try:
            df = pd.read_csv(path, sep=sep, encoding=encoding)
            print(f"  {fname}: {len(df)} samples")
            for _, r in df.iterrows():
                codes.append(str(r['code']))
                labels.append(int(r['human_label']))
                models.append(fname)
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}")

    # Also load GLM B5/B7
    try:
        df = pd.read_csv(os.path.join(BASE, 'claude_experiment', 'human_eval_claude_B5B7.csv'))
        for _, r in df.iterrows():
            codes.append(str(r['code']))
            labels.append(int(r['human_label']))
            models.append('GLM-5.1')
    except Exception as e:
        print(f"  Warning: could not read GLM B5/B7: {e}")

    # Load Sonnet
    try:
        df = pd.read_csv(os.path.join(BASE, 'sonnet_experiment', 'human_eval_sonnet_120.csv'))
        for _, r in df.iterrows():
            codes.append(str(r['code']))
            labels.append(int(r['human_label']))
            models.append('Sonnet-4-6')
    except Exception as e:
        print(f"  Warning: could not read Sonnet: {e}")

    # Load new14 GLM
    try:
        df = pd.read_csv(os.path.join(BASE, 'human_eval_glm_new14.csv'))
        for _, r in df.iterrows():
            codes.append(str(r['code']))
            labels.append(int(r['human_label']))
            models.append('GLM-5.1')
    except Exception as e:
        print(f"  Warning: could not read new14: {e}")

    print(f"Loaded {len(codes)} LLM code samples")
    return codes, labels, models


def load_isr_iterations():
    """Load ISR iteration data (initial vs refined code)."""
    isr_data = {}
    for ab in ['ISR-0', 'ISR-1', 'ISR-2', 'ISR-3']:
        path = os.path.join(BASE, 'experiment_isr', f'iterations_{ab}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            isr_data[ab] = df
            print(f"  {ab}: {len(df)} iterations from {df['prompt_id'].nunique()} prompts")
    return isr_data


def analyze_dangerous_function_distributions(vul_code, safe_code, llm_code):
    """Compare dangerous function usage across domains."""
    categories = {
        'unbounded_copy': ['strcpy', 'strcat', 'sprintf', 'gets'],
        'bounded_copy': ['strncpy', 'strncat', 'snprintf', 'memcpy', 'memmove'],
        'allocation': ['malloc', 'calloc', 'realloc', 'alloca'],
        'deallocation': ['free'],
        'string_dup': ['strdup', 'strndup'],
    }

    results = {}
    for cat_name, funcs in categories.items():
        results[cat_name] = {
            'BigVul_vul': np.mean([sum(1 for f in funcs if f in count_dangerous_calls(c)) > 0 for c in vul_code]) * 100,
            'BigVul_safe': np.mean([sum(1 for f in funcs if f in count_dangerous_calls(c)) > 0 for c in safe_code]) * 100,
            'LLM': np.mean([sum(1 for f in funcs if f in count_dangerous_calls(c)) > 0 for c in llm_code]) * 100,
        }
    return results


# ============ MAIN ============
print("=" * 60)
print("Domain Shift Root Cause Analysis")
print("=" * 60)

# 1. Load data
vul_code, safe_code, safe_changed = load_bigvul_sample()
llm_codes, llm_labels, llm_models = load_llm_code()
isr_iterations = load_isr_iterations()

# 2. Compute stats for each group
print("\nComputing code statistics...")

def batch_stats(codes, name):
    """Compute stats for a batch of code strings."""
    stats = [code_stats(c) for c in codes]
    tokens = np.array([s['tokens'] for s in stats])
    lines = np.array([s['lines'] for s in stats])
    cmt = np.array([s['comment_ratio'] for s in stats])
    total_dangerous = np.array([s['total_dangerous'] for s in stats])
    unique_dangerous = np.array([s['unique_dangerous'] for s in stats])
    return {
        'name': name, 'n': len(stats),
        'tokens_mean': np.mean(tokens), 'tokens_std': np.std(tokens),
        'lines_mean': np.mean(lines), 'lines_std': np.std(lines),
        'comment_ratio_mean': np.mean(cmt),
        'dangerous_mean': np.mean(total_dangerous), 'dangerous_std': np.std(total_dangerous),
        'unique_dangerous_mean': np.mean(unique_dangerous),
    }

bigvul_vul_stats = batch_stats(vul_code, 'BigVul Vul')
bigvul_safe_stats = batch_stats(safe_code, 'BigVul Safe')
# Separate LLM by model for comparison
deepseek_codes = [c for c, m in zip(llm_codes, llm_models) if m == 'DeepSeek-v4-pro']
glm_codes = [c for c, m in zip(llm_codes, llm_models) if m == 'GLM-5.1']
sonnet_codes = [c for c, m in zip(llm_codes, llm_models) if m == 'Sonnet-4-6']
llm_all_stats = batch_stats(llm_codes, 'LLM All')
llm_ds_stats = batch_stats(deepseek_codes, 'LLM DeepSeek') if deepseek_codes else None
llm_glm_stats = batch_stats(glm_codes, 'LLM GLM') if glm_codes else None
llm_sn_stats = batch_stats(sonnet_codes, 'LLM Sonnet') if sonnet_codes else None

# Separate LLM by human label
llm_safe_codes = [c for c, l in zip(llm_codes, llm_labels) if l == 1]
llm_unsafe_codes = [c for c, l in zip(llm_codes, llm_labels) if l == 0]
llm_safe_stats = batch_stats(llm_safe_codes, 'LLM Human-SAFE') if llm_safe_codes else None
llm_unsafe_stats = batch_stats(llm_unsafe_codes, 'LLM Human-UNSAFE') if llm_unsafe_codes else None

# 3. Print results
out_lines = []
def p(s):
    print(s)
    out_lines.append(s)

p("\n" + "=" * 60)
p("1. CODE SIZE DISTRIBUTION")
p("=" * 60)
p(f"{'Group':<25s} {'N':>5s} {'Tokens(mean)':>14s} {'Tokens(std)':>12s} {'Lines(mean)':>13s} {'Lines(std)':>11s}")
p("-" * 80)
for s in [bigvul_vul_stats, bigvul_safe_stats, llm_all_stats]:
    p(f"{s['name']:<25s} {s['n']:>5d} {s['tokens_mean']:>14.1f} {s['tokens_std']:>12.1f} {s['lines_mean']:>13.1f} {s['lines_std']:>11.1f}")
if llm_safe_stats and llm_unsafe_stats:
    for s in [llm_safe_stats, llm_unsafe_stats]:
        p(f"{s['name']:<25s} {s['n']:>5d} {s['tokens_mean']:>14.1f} {s['tokens_std']:>12.1f} {s['lines_mean']:>13.1f} {s['lines_std']:>11.1f}")

p("\nKey finding:")
p(f"  BigVul tokens CV (std/mean): vul={bigvul_vul_stats['tokens_std']/bigvul_vul_stats['tokens_mean']:.2f}, safe={bigvul_safe_stats['tokens_std']/bigvul_safe_stats['tokens_mean']:.2f}")
p(f"  LLM tokens CV: {llm_all_stats['tokens_std']/llm_all_stats['tokens_mean']:.2f}")
p(f"  → BigVul has MUCH higher variance in code length than LLM output")

p("\n" + "=" * 60)
p("2. COMMENT RATIO")
p("=" * 60)
for s in [bigvul_vul_stats, bigvul_safe_stats, llm_all_stats]:
    p(f"  {s['name']:<25s}: {s['comment_ratio_mean']:.3f} ({s['comment_ratio_mean']*100:.1f}%)")
if llm_safe_stats and llm_unsafe_stats:
    for s in [llm_safe_stats, llm_unsafe_stats]:
        p(f"  {s['name']:<25s}: {s['comment_ratio_mean']:.3f} ({s['comment_ratio_mean']*100:.1f}%)")
p("\nKey finding:")
p(f"  BigVul vul comment ratio: {bigvul_vul_stats['comment_ratio_mean']:.3f}")
p(f"  BigVul safe comment ratio: {bigvul_safe_stats['comment_ratio_mean']:.3f}")
p(f"  LLM comment ratio: {llm_all_stats['comment_ratio_mean']:.3f}")
p("  → If BigVul safe code has more comments, classifier may learn 'comments = safe'")

p("\n" + "=" * 60)
p("3. DANGEROUS FUNCTION USAGE")
p("=" * 60)
for s in [bigvul_vul_stats, bigvul_safe_stats, llm_all_stats]:
    p(f"  {s['name']:<25s}: {s['dangerous_mean']:.2f} calls/func, {s['unique_dangerous_mean']:.1f} unique types")
if llm_safe_stats and llm_unsafe_stats:
    for s in [llm_safe_stats, llm_unsafe_stats]:
        p(f"  {s['name']:<25s}: {s['dangerous_mean']:.2f} calls/func, {s['unique_dangerous_mean']:.1f} unique types")

p("\n" + "=" * 60)
p("4. DANGEROUS FUNCTION CATEGORIES (% of code using)")
p("=" * 60)
cat_results = analyze_dangerous_function_distributions(
    [extract_c_function(c) for c in vul_code[:5000]],
    [extract_c_function(c) for c in safe_code[:5000]],
    [extract_c_function(c) for c in llm_codes],
)
for cat, vals in cat_results.items():
    p(f"  {cat:<20s}: BigVul_vul={vals['BigVul_vul']:5.1f}%  BigVul_safe={vals['BigVul_safe']:5.1f}%  LLM={vals['LLM']:5.1f}%")

p("\n" + "=" * 60)
p("5. LLM CODE BY MODEL")
p("=" * 60)
for s in [llm_ds_stats, llm_glm_stats, llm_sn_stats]:
    if s:
        p(f"  {s['name']:<25s}: n={s['n']}, tokens={s['tokens_mean']:.0f}±{s['tokens_std']:.0f}, lines={s['lines_mean']:.0f}±{s['lines_std']:.0f}, cmt={s['comment_ratio_mean']:.3f}")

p("\n" + "=" * 60)
p("6. ISR ITERATION EFFECTS ON CODE STRUCTURE")
p("=" * 60)
for ab_name, df in isr_iterations.items():
    if len(df) == 0:
        continue
    # Get iteration 0 vs final iteration per prompt
    iter0 = df[df['iteration'] == 0]
    final_iters = df.groupby('prompt_id')['iteration'].max().reset_index()
    final = df.merge(final_iters, on=['prompt_id', 'iteration'])
    p(f"\n  {ab_name}:")
    p(f"    Iter 0: tokens={iter0['code_len'].mean():.0f}±{iter0['code_len'].std():.0f}, p_vul={iter0['p_vul'].mean():.6f}")
    p(f"    Final:  tokens={final['code_len'].mean():.0f}±{final['code_len'].std():.0f}, p_vul={final['p_vul'].mean():.6f}")
    p(f"    Δ tokens: {final['code_len'].mean() - iter0['code_len'].mean():.0f}, Δ p_vul: {final['p_vul'].mean() - iter0['p_vul'].mean():.6f}")

# Save
with open(OUTPUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out_lines))
print(f"\nResults saved to: {OUTPUT}")
