import csv
import json
import statistics

# Read human labels
labels = {}
with open('human_eval.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        key = (row['prompt_id'], int(row['candidate_idx']))
        labels[key] = int(row['human_label'])

# Read all_candidates
candidates = {}
with open('all_candidates.csv') as f:
    for row in csv.DictReader(f):
        key = (row['prompt_id'], int(row['candidate_idx']))
        candidates[key] = float(row['p_vul'])

# Read report
with open('report.json') as f:
    report = json.load(f)

print('=== Per-Prompt Analysis ===')
header = f"{'Prompt':<20} {'#Safe':>6} {'#Unsafe':>6} {'ClsIdx':>7} {'ClsSafe':>8} {'RndSafe':>8} {'ClsPvul':>10}"
print(header)
print('-' * len(header))

cls_safe_count = 0
rnd_safe_count = 0
prompts_with_safe = 0

for pp in report['per_prompt']:
    pid = pp['prompt_id']
    cls_pvul = pp['classifier_pick_p_vul']
    rnd_pvul = pp['random_pick_p_vul']

    safe_c = sum(1 for (p, c), l in labels.items() if p == pid and l == 1)
    unsafe_c = sum(1 for (p, c), l in labels.items() if p == pid and l == 0)

    cls_best = min([(c, candidates[(pid, c)]) for c in range(10)], key=lambda x: x[1])
    cls_idx = cls_best[0]
    cls_label = labels.get((pid, cls_idx), -1)

    rnd_idx = None
    for c in range(10):
        if abs(candidates[(pid, c)] - rnd_pvul) < 1e-10:
            rnd_idx = c
            break
    rnd_label = labels.get((pid, rnd_idx), -1) if rnd_idx is not None else -1

    if safe_c > 0:
        prompts_with_safe += 1
        if cls_label == 1:
            cls_safe_count += 1
        if rnd_label == 1:
            rnd_safe_count += 1

    print(f"{pid:<20} {safe_c:>6} {unsafe_c:>6} {cls_idx:>7} {cls_label:>8} {rnd_label:>8} {cls_pvul:>10.6f}")

print()
print('=== Summary ===')
print(f"Prompts with >=1 safe option: {prompts_with_safe}/15")
print(f"Classifier picked safe: {cls_safe_count}/{prompts_with_safe} ({cls_safe_count/prompts_with_safe*100:.1f}%)")
print(f"Random picked safe: {rnd_safe_count}/{prompts_with_safe} ({rnd_safe_count/prompts_with_safe*100:.1f}%)")

total_safe = sum(1 for l in labels.values() if l == 1)
total_unsafe = sum(1 for l in labels.values() if l == 0)
print(f"\nTotal candidates: {len(labels)}")
print(f"Safe: {total_safe} ({total_safe/len(labels)*100:.1f}%)")
print(f"Unsafe: {total_unsafe} ({total_unsafe/len(labels)*100:.1f}%)")

safe_pvuls = []
unsafe_pvuls = []
for (pid, c), label in labels.items():
    pvul = candidates[(pid, c)]
    if label == 1:
        safe_pvuls.append(pvul)
    else:
        unsafe_pvuls.append(pvul)

print(f"\nAvg P(vul) for SAFE code:   {statistics.mean(safe_pvuls):.6f}")
print(f"Avg P(vul) for UNSAFE code: {statistics.mean(unsafe_pvuls):.6f}")
print(f"Median P(vul) for SAFE:     {statistics.median(safe_pvuls):.6f}")
print(f"Median P(vul) for UNSAFE:   {statistics.median(unsafe_pvuls):.6f}")

# Per-prompt details for unsafe picks
print("\n=== Classifier Misses ===")
for pp in report['per_prompt']:
    pid = pp['prompt_id']
    cls_pvul = pp['classifier_pick_p_vul']
    cls_best = min([(c, candidates[(pid, c)]) for c in range(10)], key=lambda x: x[1])
    cls_idx = cls_best[0]
    cls_label = labels.get((pid, cls_idx), -1)
    if cls_label == 0:
        safe_count = sum(1 for (p, c), l in labels.items() if p == pid and l == 1)
        print(f"  {pid}: picked idx={cls_idx} (unsafe), {safe_count} safe alternatives available, P(vul)={cls_pvul:.6f}")
