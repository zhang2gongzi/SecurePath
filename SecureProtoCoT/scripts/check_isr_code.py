import csv
from collections import defaultdict

# Check ISR-0 data
with open('E:/paper/new/SecureProtoCoT/outputs/experiment_isr/iterations_ISR-0.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
print('ISR-0 columns:', list(rows[0].keys()))
print('Prompts:', sorted(set(r['prompt_id'] for r in rows)))

# Check ISR-3 data
with open('E:/paper/new/SecureProtoCoT/outputs/experiment_isr/iterations_ISR-3.csv', encoding='utf-8') as f:
    rows3 = list(csv.DictReader(f))

by_prompt = defaultdict(list)
for r in rows3:
    by_prompt[r['prompt_id']].append(r)

print('\n=== ISR-3 best per prompt ===')
for pid in sorted(by_prompt.keys()):
    best = min(by_prompt[pid], key=lambda r: float(r['p_vul']))
    print(f'  {pid}: best_iter={best["iteration"]}, P(vul)={float(best["p_vul"]):.6f}, code_len={best["code_len"]}')

# Show ISR-0 code samples for selected prompts
selected = ['P01_buffer_copy', 'P05_free_memory', 'P07_int_parse', 'P09_memcpy_wrapper', 'P11_struct_copy']
print('\n=== ISR-0 selected code ===')
for r in rows:
    if r['prompt_id'] in selected:
        code = r['code'][:300]
        print(f'\n--- {r["prompt_id"]} ---')
        print(code)
