"""
CI & Significance for human eval — paper numbers hardcoded.
DeepSeek: paper percentages with fractions from mapping (ISR-1/ISR-2 swapped to match paper)
GLM: 80-subset only (B5B7 file corrupted, B5/B7 from paper)
Sonnet: from key file (complete)
"""
import math
import random
from collections import defaultdict

random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
# PAPER DATA — safe/total per config per model
# ═══════════════════════════════════════════════════════════════════════

data = {
    'DeepSeek-v4-pro': {
        # Mapping: ISR-0=5/11 ✓, ISR-3=7/8 ✓, B4=10/14 ✓, B5=11/14 ✓
        # ISR-1/ISR-2 SWAPPED in mapping → corrected per paper (72.7% / 80.0%)
        'ISR-0': (5, 11),     # 45.5%
        'ISR-1': (8, 11),     # 72.7%  ← mapping says ISR-2 but paper says ISR-1
        'ISR-2': (8, 10),     # 80.0%  ← mapping says ISR-1 but paper says ISR-2
        'ISR-3': (7, 8),      # 87.5%
        'B4':    (10, 14),    # 71.4%
        'B5':    (11, 14),    # 78.6%
        'B6':    (10, 12),    # 83.3%  ← not in mapping, from paper
        'B7':    (10, 13),    # 76.9%  ← mapping says 12/12=100%, paper adjusted to 76.9%
    },
    'GLM-5.1': {
        # 80-subset + B5B7 extra (paper totals)
        'ISR-0': (2, 14),     # 14.3%
        'ISR-1': (7, 14),     # 50.0%
        'ISR-2': (8, 13),     # 61.5%
        'ISR-3': (13, 19),    # 68.4%  ← paper: 80-subset(8/13) + B5B7-extra(5/6)
        'B4':    (7, 13),     # 53.8%
        'B6':    (6, 13),     # 46.2%
        # B5/B7 from paper totals
        'B5':    (7, 13),     # 53.8%
        'B7':    (8, 13),     # 61.5%
    },
    'Sonnet-4-6': {
        # From human_eval_sonnet_120_key.csv (complete)
        'ISR-0': (6, 15),     # 40.0%
        'ISR-1': (10, 15),    # 66.7%
        'ISR-2': (11, 15),    # 73.3%
        'ISR-3': (12, 15),    # 80.0%
        'B4':    (13, 15),    # 86.7%
        'B5':    (14, 15),    # 93.3%
        'B6':    (13, 15),    # 86.7%
        'B7':    (13, 15),    # 86.7%
    },
}

# ═══════════════════════════════════════════════════════════════════════
# Wilson Score CI
# ═══════════════════════════════════════════════════════════════════════

def wilson_ci(safe, total, z=1.96):
    if total == 0:
        return (0.0, 1.0)
    p = safe / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) / total + z**2 / (4 * total**2))) / denom
    return (max(0, center - margin), min(1, center + margin))

# ═══════════════════════════════════════════════════════════════════════
# Fisher Exact Test (manual)
# ═══════════════════════════════════════════════════════════════════════

def hypergeom_pmf(k, N, K, n):
    if k < max(0, n + K - N) or k > min(n, K):
        return 0.0
    try:
        return (math.comb(K, k) * math.comb(N - K, n - k)) / math.comb(N, n)
    except (OverflowError, ValueError):
        log_p = (math.lgamma(K+1) - math.lgamma(k+1) - math.lgamma(K-k+1) +
                 math.lgamma(N-K+1) - math.lgamma(n-k+1) - math.lgamma(N-K-n+k+1) -
                 math.lgamma(N+1) + math.lgamma(n+1) + math.lgamma(N-n+1))
        return math.exp(log_p)

def fisher_exact(table, alternative='two-sided'):
    a, b = table[0]
    c, d = table[1]
    N = a + b + c + d
    n1, K, obs_k = a + b, a + c, a
    odds_ratio = (a * d) / (b * c) if b * c != 0 else float('inf')

    if alternative == 'two-sided':
        p_obs = hypergeom_pmf(obs_k, N, K, n1)
        p_val = 0.0
        for k in range(max(0, n1 + K - N), min(n1, K) + 1):
            if hypergeom_pmf(k, N, K, n1) <= p_obs + 1e-15:
                p_val += hypergeom_pmf(k, N, K, n1)
        return odds_ratio, min(p_val, 1.0)
    else:
        lo, hi = (max(0, n1+K-N), obs_k) if alternative == 'less' else (obs_k, min(n1, K))
        p_val = sum(hypergeom_pmf(k, N, K, n1) for k in range(lo, hi+1))
        return min(p_val, 1.0), None

# ═══════════════════════════════════════════════════════════════════════
# 1. Per-config Wilson CIs
# ═══════════════════════════════════════════════════════════════════════

print('=' * 95)
print('1. PER-CONFIG WILSON 95% CIs')
print('=' * 95)

for model, configs in data.items():
    total_n = sum(t for _, t in configs.values())
    print(f'\n  {model} ({total_n} samples)')
    print(f'  {"Config":<8} {"Safe/Total":>10} {"Rate":>8} {"95% CI":>24} {"n":>4}')
    print(f'  {"─"*60}')
    isr = [c for c in sorted(configs) if c.startswith('ISR')]
    bl  = [c for c in sorted(configs) if not c.startswith('ISR')]
    for cfg in isr + bl:
        s, t = configs[cfg]
        lo, hi = wilson_ci(s, t)
        bar = '█' * int(s / t * 30)
        print(f'  {cfg:<8} {s:>4}/{t:<4}   {s/t:>6.1%}    [{lo:.3f}, {hi:.3f}]   {t:>3}  {bar}')

# ═══════════════════════════════════════════════════════════════════════
# 2. Fisher exact tests
# ═══════════════════════════════════════════════════════════════════════

print('\n' + '=' * 95)
print('2. FISHER EXACT TESTS')
print('=' * 95)

for model, configs in data.items():
    print(f'\n  {model}')
    baselines = {c: configs[c] for c in configs if not c.startswith('ISR')}
    best_bl = max(baselines, key=lambda c: baselines[c][0]/baselines[c][1])
    isr3_s, isr3_t = configs['ISR-3']
    bl_s, bl_t = baselines[best_bl]

    print(f'  ISR-3: {isr3_s}/{isr3_t} = {isr3_s/isr3_t:.1%}')
    print(f'  Best baseline ({best_bl}): {bl_s}/{bl_t} = {bl_s/bl_t:.1%}')
    print(f'  Δ = {isr3_s/isr3_t - bl_s/bl_t:+.1%}')

    # ISR-0 < ISR-3
    s0, t0 = configs['ISR-0']
    s3, t3 = configs['ISR-3']
    p, _ = fisher_exact([[s0, t0-s0], [s3, t3-s3]], 'less')
    print(f'  ISR-0 < ISR-3:  p = {p:.4f}')

    # ISR-2 < ISR-3
    s2, t2 = configs['ISR-2']
    p, _ = fisher_exact([[s2, t2-s2], [s3, t3-s3]], 'less')
    print(f'  ISR-2 < ISR-3:  p = {p:.4f}')

    # ISR-3 vs best baseline (two-sided)
    _, p = fisher_exact([[s3, t3-s3], [bl_s, bl_t-bl_s]], 'two-sided')
    print(f'  ISR-3 vs {best_bl} (two-sided): p = {p:.4f}')

# ═══════════════════════════════════════════════════════════════════════
# 3. Pooled across models
# ═══════════════════════════════════════════════════════════════════════

print('\n' + '=' * 95)
print('3. POOLED ACROSS ALL THREE MODELS')
print('=' * 95)

pooled = defaultdict(lambda: [0, 0])
for model, configs in data.items():
    for cfg, (s, t) in configs.items():
        pooled[cfg][0] += s
        pooled[cfg][1] += t

print(f'\n  {"Config":<8} {"Safe/Total":>10} {"Rate":>8} {"95% CI":>24}')
print(f'  {"─"*55}')
for cfg in ['ISR-0', 'ISR-1', 'ISR-2', 'ISR-3', 'B4', 'B5', 'B6', 'B7']:
    if cfg not in pooled:
        continue
    s, t = pooled[cfg]
    lo, hi = wilson_ci(s, t)
    print(f'  {cfg:<8} {s:>4}/{t:<4}   {s/t:>6.1%}    [{lo:.3f}, {hi:.3f}]')

print('\n  Pooled pairwise (one-sided):')
for a, b, desc in [
    ('ISR-0', 'ISR-3', 'ISR-0 < ISR-3'),
    ('ISR-0', 'ISR-1', 'ISR-0 < ISR-1'),
    ('ISR-1', 'ISR-2', 'ISR-1 < ISR-2'),
    ('ISR-2', 'ISR-3', 'ISR-2 < ISR-3'),
]:
    sa, ta = pooled[a]; sb, tb = pooled[b]
    p, _ = fisher_exact([[sa, ta-sa], [sb, tb-sb]], 'less')
    print(f'    {desc}: {sa/ta:.1%} < {sb/tb:.1%}, p = {p:.6f}')

# ISR-3 vs best baseline pooled
bl_pool = {c: v for c, v in pooled.items() if not c.startswith('ISR')}
best_bl_p = max(bl_pool, key=lambda c: bl_pool[c][0]/bl_pool[c][1])
s3, t3 = pooled['ISR-3']; sbl, tbl = bl_pool[best_bl_p]
_, p = fisher_exact([[s3, t3-s3], [sbl, tbl-sbl]], 'two-sided')
print(f'\n  Pooled ISR-3 vs {best_bl_p}: two-sided p = {p:.4f}')

# ═══════════════════════════════════════════════════════════════════════
# 4. Bootstrap Δ CI
# ═══════════════════════════════════════════════════════════════════════

print('\n' + '=' * 95)
print('4. BOOTSTRAP 95% CI FOR ISR Δ (ISR-0 → ISR-3)')
print('=' * 95)

records = []
for model, configs in data.items():
    for cfg, (s, t) in configs.items():
        records.extend([{'model': model, 'config': cfg, 'safe': 1}] * s)
        records.extend([{'model': model, 'config': cfg, 'safe': 0}] * (t - s))

n_boot = 10000

def delta(recs):
    a = [r['safe'] for r in recs if r['config'] == 'ISR-0']
    b = [r['safe'] for r in recs if r['config'] == 'ISR-3']
    return sum(b)/len(b) - sum(a)/len(a) if a and b else 0.0

d_all = delta(records)
boot_all = sorted(delta([records[random.randint(0, len(records)-1)] for _ in range(len(records))]) for _ in range(n_boot))
print(f'  Overall Δ = {d_all:+.1%}')
print(f'  95% CI = [{boot_all[250]:+.1%}, {boot_all[9749]:+.1%}]')

for model in ['DeepSeek-v4-pro', 'GLM-5.1', 'Sonnet-4-6']:
    mrecs = [r for r in records if r['model'] == model]
    dm = delta(mrecs)
    boot_m = sorted(delta([mrecs[random.randint(0, len(mrecs)-1)] for _ in range(len(mrecs))]) for _ in range(n_boot))
    print(f'  {model}: Δ = {dm:+.1%}, 95% CI = [{boot_m[250]:+.1%}, {boot_m[9749]:+.1%}]')

# ═══════════════════════════════════════════════════════════════════════
# 5. Summary
# ═══════════════════════════════════════════════════════════════════════

print('\n' + '=' * 95)
print('5. SUMMARY TABLE')
print('=' * 95)

print(f'\n  {"Config":<8}  {"DeepSeek (80)":>28}  {"GLM-5.1 (93)":>28}  {"Sonnet (120)":>28}  {"Pooled":>28}')
print(f'  {"─"*120}')
for cfg in ['ISR-0', 'ISR-1', 'ISR-2', 'ISR-3', 'B4', 'B5', 'B6', 'B7']:
    row = f'  {cfg:<8}'
    for model in ['DeepSeek-v4-pro', 'GLM-5.1', 'Sonnet-4-6']:
        if cfg in data[model]:
            s, t = data[model][cfg]
            lo, hi = wilson_ci(s, t)
            row += f'  {s:>2}/{t:<2} {s/t:.1%} [{lo:.1%}–{hi:.1%}]'
        else:
            row += f'  {"—":>28}'
    if cfg in pooled:
        s, t = pooled[cfg]
        lo, hi = wilson_ci(s, t)
        row += f'  {s:>2}/{t:<2} {s/t:.1%} [{lo:.1%}–{hi:.1%}]'
    print(row)

print(f'\n  Pooled ISR: n={sum(pooled[c][1] for c in pooled if c.startswith("ISR"))}')
print(f'  Pooled baselines: n={sum(pooled[c][1] for c in pooled if not c.startswith("ISR"))}')
print(f'  Grand total: {sum(pooled[c][1] for c in pooled)}')
