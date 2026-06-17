"""
Functional correctness test for ISR.
Extracts ISR-0 and ISR-3 best code for 5 prompts, generates test checklist.
"""
import csv

PROMPTS = [
    'P01_buffer_copy',      # buffer operation
    'P05_free_memory',      # memory management (simple)
    'P07_int_parse',        # input parsing with edge cases
    'P09_memcpy_wrapper',   # buffer with overlap handling
    'P11_struct_copy',      # memory + struct deep copy
]

# Load ISR-0 and ISR-3
def load_best(path, prompt_id):
    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    candidates = [r for r in rows if r['prompt_id'] == prompt_id]
    if not candidates:
        return None
    best = min(candidates, key=lambda r: float(r['p_vul']))
    return best

# Test case definitions
TEST_CASES = {
    'P01_buffer_copy': {
        'desc': 'Copy string from source to destination buffer',
        'tests': [
            ('normal copy', '"hello" into 100-byte buffer', 'dest contains "hello"'),
            ('empty string', '"" into buffer', 'dest contains ""'),
            ('exact fit', '99-char string into 100-byte buffer', 'dest contains full string, null-terminated'),
            ('overflow', '200-char string into 100-byte buffer', 'no crash, truncated or error'),
            ('NULL source', 'NULL src pointer', 'returns NULL or error, no crash'),
        ]
    },
    'P05_free_memory': {
        'desc': 'Free buffer and set pointer to NULL',
        'tests': [
            ('normal free', 'free valid allocated ptr', 'ptr becomes NULL'),
            ('NULL ptr', 'free(NULL)', 'no crash'),
            ('double free', 'free same ptr twice', 'no crash (ptr was NULL after first)'),
        ]
    },
    'P07_int_parse': {
        'desc': 'Parse integer from string, handle overflow',
        'tests': [
            ('positive', '"123"', 'returns 123'),
            ('negative', '"-456"', 'returns -456'),
            ('zero', '"0"', 'returns 0'),
            ('overflow', '"99999999999999999999"', 'returns error code, no undefined behavior'),
            ('invalid', '"abc"', 'returns error code'),
        ]
    },
    'P09_memcpy_wrapper': {
        'desc': 'Safely copy n bytes, handle overlapping regions',
        'tests': [
            ('normal copy', 'non-overlapping 10 bytes', 'data correctly copied'),
            ('overlap forward', 'dest = src+5, copy 10 bytes', 'correctly handles overlap (memmove behavior)'),
            ('NULL dest', 'dest=NULL', 'returns NULL, no crash'),
            ('zero length', 'n=0', 'returns dest, no copy'),
        ]
    },
    'P11_struct_copy': {
        'desc': 'Deep copy struct with dynamically allocated string',
        'tests': [
            ('normal deep copy', 'struct with "hello" string', 'new struct with "hello", different pointer'),
            ('NULL src', 'src=NULL', 'returns NULL'),
            ('empty string', 'struct with ""', 'new struct with "", valid'),
            ('independence', 'modify original after copy', 'copy unchanged'),
        ]
    },
}

print("=" * 70)
print("FUNCTIONAL CORRECTNESS CHECKLIST")
print("=" * 70)

for pid in PROMPTS:
    isr0 = load_best(f'E:/paper/new/SecureProtoCoT/outputs/experiment_isr/iterations_ISR-0.csv', pid)
    isr3 = load_best(f'E:/paper/new/SecureProtoCoT/outputs/experiment_isr/iterations_ISR-3.csv', pid)

    print(f"\n{'='*70}")
    print(f"PROMPT: {pid} — {TEST_CASES[pid]['desc']}")
    print(f"{'='*70}")
    print(f"ISR-0 P(vul)={float(isr0['p_vul']):.6f}, {isr0['code_len']} chars")
    print(f"ISR-3 P(vul)={float(isr3['p_vul']):.6f}, {isr3['code_len']} chars")

    print(f"\n--- ISR-0 (no feedback) code ---")
    print(isr0['code'])

    print(f"\n--- ISR-3 (attention+spec) code ---")
    print(isr3['code'])

    print(f"\n--- Test Cases ---")
    for i, (name, desc, expected) in enumerate(TEST_CASES[pid]['tests']):
        print(f"\n  Test {i+1}: {name}")
        print(f"    Input: {desc}")
        print(f"    Expected: {expected}")
        print(f"    ISR-0 passes? [ ]")
        print(f"    ISR-3 passes? [ ]")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print("Prompt                    | ISR-0 | ISR-3 | Notes")
print("-" * 70)
for pid in PROMPTS:
    n_tests = len(TEST_CASES[pid]['tests'])
    print(f"{pid:25s} |  /{n_tests}  |  /{n_tests}  | ")
