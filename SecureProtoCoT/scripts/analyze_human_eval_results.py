import csv
from collections import defaultdict

with open("E:/paper/new/SecureProtoCoT/outputs/human_eval_blinded.csv", encoding='gb18030') as f:
    reader = csv.DictReader(f, delimiter='\t')
    rows = list(reader)

labels = [(int(r['eval_id']), int(r['human_label']), r['issue'][:60] if r['issue'].strip() else '')
          for r in rows if r['human_label'].strip()]

total = len(rows)
done = len(labels)
safe = sum(1 for _, l, _ in labels if l == 1)
unsafe = sum(1 for _, l, _ in labels if l == 0)
missing = [r['eval_id'] for r in rows if not r['human_label'].strip()]

print(f"Total: {total}, Done: {done}, SAFE: {safe}, UNSAFE: {unsafe}")
if missing:
    print(f"Missing IDs: {missing}")
else:
    print("All 80 done!")

with open("E:/paper/new/SecureProtoCoT/outputs/human_eval_mapping.csv", encoding='utf-8-sig') as f:
    mapping = {row['eval_id']: row for row in csv.DictReader(f)}

by_source = defaultdict(lambda: {'total': 0, 'safe': 0})
by_pvul_band = defaultdict(lambda: {'total': 0, 'safe': 0})
by_prompt = defaultdict(lambda: {'total': 0, 'safe': 0})

for r in rows:
    eid = r['eval_id']
    if not r['human_label'].strip():
        continue
    label = int(r['human_label'])
    src = mapping[eid]['source']
    pid = mapping[eid]['prompt_id']
    pv = float(mapping[eid]['p_vul'])

    by_source[src]['total'] += 1
    by_source[src]['safe'] += label

    by_prompt[pid]['total'] += 1
    by_prompt[pid]['safe'] += label

    if pv < 0.001:
        band = '<0.001'
    elif pv < 0.1:
        band = '0.001-0.1'
    else:
        band = '>0.1'
    by_pvul_band[band]['total'] += 1
    by_pvul_band[band]['safe'] += label

print("\n=== By Source ===")
for src in sorted(by_source.keys()):
    d = by_source[src]
    rate = d['safe']/d['total']*100 if d['total'] else 0
    print(f"  {src}: {d['safe']}/{d['total']} ({rate:.1f}%)")

print("\n=== By P(vul) Band ===")
for band in ['<0.001', '0.001-0.1', '>0.1']:
    d = by_pvul_band[band]
    rate = d['safe']/d['total']*100 if d['total'] else 0
    print(f"  P(vul) {band}: {d['safe']}/{d['total']} ({rate:.1f}%)")

pvuls = []
hlabels = []
for r in rows:
    eid = r['eval_id']
    if not r['human_label'].strip():
        continue
    pvuls.append(float(mapping[eid]['p_vul']))
    hlabels.append(int(r['human_label']))

try:
    from scipy.stats import spearmanr
    corr, pval = spearmanr(pvuls, hlabels)
    print(f"\nSpearman r = {corr:.4f}, p = {pval:.4f}")
except ImportError:
    # manual spearman
    n = len(pvuls)
    rank_pv = {v: i+1 for i, v in enumerate(sorted(set(pvuls)))}
    rank_hl = {v: i+1 for i, v in enumerate(sorted(set(hlabels)))}
    # simplified: just use pearson on ranks
    import statistics
    mean_rp = sum(rank_pv[v] for v in pvuls) / n
    mean_rh = sum(rank_hl[v] for v in hlabels) / n
    num = sum((rank_pv[pvuls[i]] - mean_rp) * (rank_hl[hlabels[i]] - mean_rh) for i in range(n))
    den = (sum((rank_pv[v] - mean_rp)**2 for v in pvuls) * sum((rank_hl[v] - mean_rh)**2 for v in hlabels)) ** 0.5
    if den > 0:
        corr = num / den
    else:
        corr = 0
    print(f"\nSpearman r (manual) = {corr:.4f}")

isr_sources = {'ISR-0', 'ISR-1', 'ISR-2', 'ISR-3'}
baseline_sources = {'B4_SafePrompt', 'B5_SVEN', 'B7_CoSec'}

isr_tot = sum(by_source[s]['total'] for s in isr_sources)
isr_safe = sum(by_source[s]['safe'] for s in isr_sources)
bl_tot = sum(by_source[s]['total'] for s in baseline_sources)
bl_safe = sum(by_source[s]['safe'] for s in baseline_sources)

print(f"\n=== ISR vs Baseline (aggregate) ===")
print(f"  ISR: {isr_safe}/{isr_tot} ({isr_safe/isr_tot*100:.1f}%)")
print(f"  Baseline (B4+B5+B7): {bl_safe}/{bl_tot} ({bl_safe/bl_tot*100:.1f}%)")

# Per-baseline safe rates
print("\n=== Per-Baseline Safe Rate ===")
for src in ['B4_SafePrompt', 'B5_SVEN', 'B7_CoSec']:
    d = by_source[src]
    rate = d['safe']/d['total']*100 if d['total'] else 0
    print(f"  {src}: {d['safe']}/{d['total']} ({rate:.1f}%)")

# ISR per ablation
print("\n=== ISR Per-Ablation ===")
for src in ['ISR-0', 'ISR-1', 'ISR-2', 'ISR-3']:
    d = by_source[src]
    rate = d['safe']/d['total']*100 if d['total'] else 0
    print(f"  {src}: {d['safe']}/{d['total']} ({rate:.1f}%)")

# Prompts with most issues
print("\n=== By Prompt (SAFE rate) ===")
for pid in sorted(by_prompt.keys()):
    d = by_prompt[pid]
    rate = d['safe']/d['total']*100 if d['total'] else 0
    print(f"  {pid}: {d['safe']}/{d['total']} ({rate:.1f}%)")
