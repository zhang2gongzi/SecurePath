"""
Score all LLM-generated code snippets using the safety classifier on the server.
Supports GLM (claude_experiment) and Sonnet (sonnet_experiment) directories.

Usage:
    cd /home2/zzl/SecurePath/SecureProtoCoT/scripts
    python score_claude_experiment.py ../outputs/claude_experiment    # GLM-5.1
    python score_claude_experiment.py ../outputs/sonnet_experiment    # Sonnet-4-6
"""

import re
import csv
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModel
from attention_pooling import SafetyClassifier

# --- Config ---
CONFIG = {
    'encoder_path': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/models/best_model',
    'classifier_path': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/models/safety_classifier_attn.pt',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

CONFIGS = ["ISR0", "ISR1", "ISR2", "ISR3", "B4", "B5", "B6", "B7"]


def extract_codes_from_md(filepath):
    """Extract all code snippets from a markdown file.
    Returns list of (prompt_id, candidate_id, code_text) tuples.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    entries = []
    prompt_sections = re.split(r'^## (P\d+_\w+)', content, flags=re.MULTILINE)

    for i in range(1, len(prompt_sections), 2):
        prompt_key = prompt_sections[i]
        prompt_id = prompt_key.split("_")[0]
        section_body = prompt_sections[i + 1]

        cand_sections = re.split(r'^### (c\d+)', section_body, flags=re.MULTILINE)

        for j in range(1, len(cand_sections), 2):
            cand_id = cand_sections[j]
            cand_body = cand_sections[j + 1]

            code_match = re.search(r'```c\n(.*?)```', cand_body, re.DOTALL)
            if code_match:
                code_text = code_match.group(1).strip()
                # Remove iteration annotation line from B6
                code_text = re.sub(r'^\s*//.*iter.*$', '', code_text, flags=re.MULTILINE).strip()
                # Normalize: collapse excessive blank lines
                code_text = re.sub(r'\n{3,}', '\n\n', code_text)
                entries.append((prompt_id, cand_id, code_text))

    return entries


@torch.no_grad()
def score_code(code, tokenizer, codebert, classifier):
    """Return P(vul) for a single code snippet."""
    device = CONFIG['device']
    enc = tokenizer(code, max_length=512, padding='max_length',
                    truncation=True, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    outputs = codebert(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = outputs.last_hidden_state
    logits = classifier(hidden_states, attention_mask)
    p_vul = torch.nn.functional.softmax(logits, dim=1)[0, 1].item()
    return p_vul


def main():
    if len(sys.argv) < 2:
        print("Usage: python score_claude_experiment.py <data_dir>")
        print("  data_dir: path containing ISR0/ ISR1/ ISR2/ ISR3/ B4/ B6/ subdirs")
        sys.exit(1)

    data_dir = sys.argv[1]

    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['encoder_path'])
    codebert = AutoModel.from_pretrained(CONFIG['encoder_path']).to(CONFIG['device'])
    codebert.eval()
    classifier = SafetyClassifier().to(CONFIG['device'])
    classifier.load_state_dict(torch.load(CONFIG['classifier_path'], map_location=CONFIG['device']))
    classifier.eval()
    print(f"Models loaded on {CONFIG['device']}")

    results = []
    total = 0

    for config in CONFIGS:
        for batch in ["P01_P05", "P06_P10", "P11_P15"]:
            filepath = os.path.join(data_dir, config, f"{config}_{batch}.md")
            if not os.path.exists(filepath):
                print(f"WARNING: {filepath} not found, skipping")
                continue

            entries = extract_codes_from_md(filepath)
            for prompt_id, cand_id, code_text in entries:
                p_vul = score_code(code_text, tokenizer, codebert, classifier)
                results.append({
                    'config': config,
                    'prompt_id': prompt_id,
                    'candidate_id': cand_id,
                    'p_vul': p_vul,
                })
                total += 1
                if total % 50 == 0:
                    print(f"  Scored {total}...")

    print(f"Scored {total} snippets total.")

    # Save
    output_path = os.path.join(data_dir, "claude_pvul_scores.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=['config', 'prompt_id', 'candidate_id', 'p_vul'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved to {output_path}")

    # Quick summary
    from collections import defaultdict
    cfg_stats = defaultdict(list)
    for r in results:
        cfg_stats[r['config']].append(r['p_vul'])
    print("\nPer-config P(vul) summary:")
    print(f"{'Config':<8} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10}")
    for cfg in CONFIGS:
        vals = cfg_stats[cfg]
        print(f"{cfg:<8} {sum(vals)/len(vals):>10.6f} {sorted(vals)[len(vals)//2]:>10.6f} {min(vals):>10.6f} {max(vals):>10.6f}")


if __name__ == "__main__":
    main()