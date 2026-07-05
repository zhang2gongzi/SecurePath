"""
Extract code snippets from Claude experiment markdown files and create
a stratified 80-sample blinded evaluation sheet.

Sampling strategy (no P(vul) scores available):
- 6 configs, each gets ~13-14 samples (total 80)
- Within each config: spread across 15 prompts, random candidate selection
- Output: CSV with eval_id, prompt_desc, code, blank human_label, blank issue
"""

import re
import csv
import random
import os

random.seed(20260602)

CONFIGS = ["ISR0", "ISR1", "ISR2", "ISR3", "B4", "B6"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Prompt descriptions
PROMPT_DESCS = {
    "P01": "Copy a string from src to dst with size limit (buffer copy safety)",
    "P02": "Access array element by index with bounds check",
    "P03": "Allocate and zero-initialize an integer array",
    "P04": "Concatenate two strings into a newly allocated buffer",
    "P05": "Free a dynamically allocated buffer and null the pointer",
    "P06": "Read entire file contents into a dynamically allocated buffer",
    "P07": "Parse an integer from a string with overflow detection",
    "P08": "Insert a node at the head of a singly linked list",
    "P09": "Copy memory from src to dst with overlap handling",
    "P10": "Read lines from stdin in a loop until 'quit' is entered",
    "P11": "Deep copy a struct containing a dynamically allocated string",
    "P12": "Recursively free a binary tree with string data fields",
    "P13": "Format a log message with timestamp, level, and message text",
    "P14": "Double the size of a dynamic array using realloc",
    "P15": "Parse --output flag from command-line arguments",
}

def extract_codes_from_md(filepath):
    """Extract all code snippets from a markdown file.
    Returns list of (prompt_id, candidate_id, code_text) tuples.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    entries = []
    # Split by ## headers for prompt sections
    prompt_sections = re.split(r'^## (P\d+_\w+)', content, flags=re.MULTILINE)

    # prompt_sections: [prelude, P01_xxx, body1, P02_xxx, body2, ...]
    for i in range(1, len(prompt_sections), 2):
        prompt_key = prompt_sections[i]  # e.g. "P01_buffer_copy"
        prompt_id = prompt_key.split("_")[0]  # e.g. "P01"
        section_body = prompt_sections[i + 1]

        # Split by ### headers for candidate sections
        cand_sections = re.split(r'^### (c\d+)', section_body, flags=re.MULTILINE)

        for j in range(1, len(cand_sections), 2):
            cand_id = cand_sections[j]  # e.g. "c0"
            cand_body = cand_sections[j + 1]

            # Extract code between ```c and ```
            code_match = re.search(r'```c\n(.*?)```', cand_body, re.DOTALL)
            if code_match:
                code_text = code_match.group(1).strip()
                # Remove iteration annotation line from B6 if present
                code_text = re.sub(r'^\s*//.*iter.*$', '', code_text, flags=re.MULTILINE).strip()
                entries.append((prompt_id, cand_id, code_text))

    return entries


def main():
    # Step 1: Parse all 18 markdown files
    all_codes = {}  # config -> prompt_id -> {cand_id: code}

    for config in CONFIGS:
        all_codes[config] = {}
        for batch in ["P01_P05", "P06_P10", "P11_P15"]:
            filepath = os.path.join(BASE_DIR, config, f"{config}_{batch}.md")
            entries = extract_codes_from_md(filepath)
            for prompt_id, cand_id, code_text in entries:
                if prompt_id not in all_codes[config]:
                    all_codes[config][prompt_id] = {}
                all_codes[config][prompt_id][cand_id] = code_text

    # Verify counts
    total = 0
    for config in CONFIGS:
        count = sum(len(cands) for cands in all_codes[config].values())
        print(f"{config}: {len(all_codes[config])} prompts, {count} candidates")
        total += count
    print(f"Total: {total}")

    # Step 2: Stratified sampling - 80 samples across 6 configs
    # Each config gets 13 or 14 samples
    samples_per_config = {c: 14 for c in ["ISR0", "ISR1", "ISR2", "ISR3"]}
    samples_per_config["B4"] = 12
    samples_per_config["B6"] = 12
    # ISR0=14, ISR1=14, ISR2=14, ISR3=14, B4=12, B6=12 = 80

    sampled = []

    # Strategy: ensure each prompt appears at least 5 times across 6 configs
    # 15 prompts × 5 minimum = 75, leaving 5 extra to distribute
    # Assign extra to prompts that show the most variance (P01, P07, P09, P14, P15)

    MIN_PER_PROMPT = 5
    EXTRA_PROMPTS = ["P01", "P07", "P09", "P14", "P15"]  # +1 each = 80 total

    # For each prompt, decide which configs contribute
    # Round-robin: each prompt gets MIN_PER_PROMPT configs, one sample each
    # Then EXTRA_PROMPTS get one additional config

    # Build assignment: prompt -> list of (config, n_samples)
    # We need to balance configs too: each config should get ~13-14
    config_counts = {c: 0 for c in CONFIGS}

    # First pass: assign 1 sample per prompt per config, cycling through configs
    for p in sorted(PROMPT_DESCS.keys()):
        # Assign this prompt to MIN_PER_PROMPT configs (least used so far)
        available = sorted(CONFIGS, key=lambda c: config_counts[c])
        assigned_configs = available[:MIN_PER_PROMPT]
        for c in assigned_configs:
            if p in all_codes[c]:
                candidates = sorted(all_codes[c][p].keys())
                selected_cands = random.sample(candidates, 1)
                for sc in selected_cands:
                    sampled.append({
                        "config": c,
                        "prompt_id": p,
                        "candidate_id": sc,
                        "prompt_desc": PROMPT_DESCS[p],
                        "code": all_codes[c][p][sc],
                    })
                    config_counts[c] += 1

    # Second pass: add 1 extra sample for EXTRA_PROMPTS
    for p in EXTRA_PROMPTS:
        # Find the least-used config that has this prompt
        available = sorted(CONFIGS, key=lambda c: config_counts[c])
        for c in available:
            if p in all_codes[c] and config_counts[c] < 14:
                candidates = sorted(all_codes[c][p].keys())
                # Avoid picking same candidate already sampled
                already = set(s["candidate_id"] for s in sampled
                              if s["config"] == c and s["prompt_id"] == p)
                remaining = [x for x in candidates if x not in already]
                if remaining:
                    sc = random.choice(remaining)
                    sampled.append({
                        "config": c,
                        "prompt_id": p,
                        "candidate_id": sc,
                        "prompt_desc": PROMPT_DESCS[p],
                        "code": all_codes[c][p][sc],
                    })
                    config_counts[c] += 1
                    break

    # Shuffle and assign eval_id
    random.shuffle(sampled)
    for i, s in enumerate(sampled):
        s["eval_id"] = i + 1

    # Step 3: Write blinded CSV (no config/candidate info visible)
    output_path = os.path.join(BASE_DIR, "human_eval_claude_80.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_id", "prompt_desc", "code", "human_label", "issue"])
        for s in sorted(sampled, key=lambda x: x["eval_id"]):
            writer.writerow([
                s["eval_id"],
                s["prompt_desc"],
                s["code"],
                "",  # blank for evaluator
                "",  # blank for evaluator
            ])

    # Step 4: Write key file (hidden, for post-eval merge)
    key_path = os.path.join(BASE_DIR, "human_eval_claude_80_key.csv")
    with open(key_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_id", "config", "prompt_id", "candidate_id"])
        for s in sorted(sampled, key=lambda x: x["eval_id"]):
            writer.writerow([
                s["eval_id"],
                s["config"],
                s["prompt_id"],
                s["candidate_id"],
            ])

    print(f"\nWritten {len(sampled)} samples:")
    print(f"  Blinded CSV: {output_path}")
    print(f"  Key file:    {key_path}")
    print(f"\nPer-config distribution:")
    for config in CONFIGS:
        n = sum(1 for s in sampled if s["config"] == config)
        print(f"  {config}: {n}")
    print(f"\nPer-prompt distribution:")
    for p in sorted(PROMPT_DESCS.keys()):
        n = sum(1 for s in sampled if s["prompt_id"] == p)
        print(f"  {p}: {n}")


if __name__ == "__main__":
    main()