#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按CWE类型分组评估分类器性能，输出per-CWE指标表
"""
import os, sys, torch, numpy as np, pandas as pd
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from attention_pooling import SafetyClassifier

# ================= 配置 =================
CONFIG = {
    'encoder_path': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/models/best_model',
    'classifier_path': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/models/safety_classifier_attn.pt',
    'data_dir': '/home2/zzl/SecurePath/SecureProtoCoT/data/processed',
    'max_length': 512,
    'batch_size': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

@torch.no_grad()
def predict_batch(codes, codebert, classifier, tokenizer, device, batch_size):
    probs = []
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        enc = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = codebert(input_ids=enc['input_ids'].to(device),
                          attention_mask=enc['attention_mask'].to(device))
        hidden_states = outputs.last_hidden_state  # (B, L, 768)
        logits = classifier(hidden_states, enc['attention_mask'].to(device))
        p = torch.nn.functional.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs.extend(p.tolist())
    return np.array(probs)

def main():
    print("=" * 60)
    print("Per-CWE 安全判别能力分析")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['encoder_path'])
    codebert = AutoModel.from_pretrained(CONFIG['encoder_path']).to(CONFIG['device'])
    codebert.eval()

    classifier = SafetyClassifier().to(CONFIG['device'])
    classifier.load_state_dict(torch.load(CONFIG['classifier_path'], map_location=CONFIG['device']))
    classifier.eval()

    # 加载测试集（含cwe_type）
    test_df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'test_contrastive.csv'))
    codes = test_df['code'].astype(str).tolist()
    labels = test_df['label'].astype(int).values
    cwe_types = test_df['cwe_type'].values

    print(f"\n总样本: {len(test_df)}")
    print("按CWE分布:")
    for cwe in sorted(set(cwe_types)):
        cwe_mask = cwe_types == cwe
        print(f"  {cwe}: {cwe_mask.sum()} 条 (vul={labels[cwe_mask].sum()}, safe={sum(1 for l in labels[cwe_mask] if l==0)})")

    # 预测
    print("\n预测中...")
    p_vul = predict_batch(codes, codebert, classifier, tokenizer, CONFIG['device'], CONFIG['batch_size'])
    preds = (p_vul >= 0.5).astype(int)

    # Per-CWE 评估
    print(f"\n{'=' * 80}")
    print(f"{'CWE':<12} {'样本数':>6} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 80)

    rows = []
    all_labels, all_preds, all_probs = [], [], []
    for cwe in sorted(set(cwe_types)):
        mask = cwe_types == cwe
        n = mask.sum()
        y_true = labels[mask]
        y_pred = preds[mask]
        y_prob = p_vul[mask]

        if len(set(y_true)) < 2:
            auc_val = float('nan')
        else:
            auc_val = roc_auc_score(y_true, y_prob)

        row = {
            'CWE': cwe,
            'N': n,
            'Acc': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'AUC': auc_val,
        }
        rows.append(row)
        print(f"{cwe:<12} {n:>6} {row['Acc']:>8.4f} {row['Precision']:>8.4f} {row['Recall']:>8.4f} {row['F1']:>8.4f} {str(row['AUC']):>8}")

        all_labels.extend(y_true)
        all_preds.extend(y_pred)
        all_probs.extend(y_prob)

    # 总体
    print("-" * 80)
    total_row = {
        'CWE': 'ALL',
        'N': len(all_labels),
        'Acc': accuracy_score(all_labels, all_preds),
        'Precision': precision_score(all_labels, all_preds, zero_division=0),
        'Recall': recall_score(all_labels, all_preds, zero_division=0),
        'F1': f1_score(all_labels, all_preds, zero_division=0),
        'AUC': roc_auc_score(all_labels, all_probs),
    }
    print(f"{'总体':<12} {total_row['N']:>6} {total_row['Acc']:>8.4f} {total_row['Precision']:>8.4f} {total_row['Recall']:>8.4f} {total_row['F1']:>8.4f} {total_row['AUC']:>8.4f}")

    # 保存
    pd.DataFrame(rows + [total_row]).to_csv(
        os.path.join(CONFIG['data_dir'], '/home2/zzl/SecurePath/SecureProtoCoT/outputs/results/per_cwe_metrics.csv'), index=False)
    print(f"\n结果已保存: outputs/results/per_cwe_metrics.csv")


if __name__ == '__main__':
    main()
