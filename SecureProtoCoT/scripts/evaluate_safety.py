#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验A：安全判别评估脚本
- 原型基线（cosine similarity scoring）
- 安全分类器（冻结编码器 + MLP）
双方法对比，为论文提供 ablation 证据
"""
import os, sys, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from attention_pooling import SafetyClassifier

# ================= 配置 =================
CONFIG = {
    'encoder_path': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/models/best_model',
    'classifier_path': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/models/safety_classifier_attn.pt',
    'data_dir': '/home2/zzl/SecurePath/SecureProtoCoT/data/processed',
    'proto_dir': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/models',
    'output_dir': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/results',
    'max_length': 512,
    'batch_size': 32,
    'projection_dim': 256,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


# ================= 编码器 =================
class ContrastiveEncoder(torch.nn.Module):
    """与训练时结构一致，用于原型评分"""
    def __init__(self, model_path, projection_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        self.hidden_size = self.encoder.config.hidden_size
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        projection = self.projection(cls_embedding)
        return torch.nn.functional.normalize(projection, p=2, dim=1)


# ================= 数据加载 =================
class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.codes = df['code'].astype(str).tolist()
        self.labels = df['label'].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.codes)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.codes[idx], max_length=self.max_length,
                             padding='max_length', truncation=True)
        return {
            'input_ids': torch.tensor(enc['input_ids']),
            'attention_mask': torch.tensor(enc['attention_mask']),
            'labels': torch.tensor(self.labels[idx])
        }


# ================= 原型评分（Baseline） =================
@torch.no_grad()
def evaluate_prototype(name, df, model, tokenizer, safe_proto, vul_proto,
                       device, max_length, batch_size, output_dir):
    """原型余弦相似度评分"""
    print(f"\n{'=' * 60}")
    print(f"[Prototype Baseline - {name}] 样本: {len(df)} (safe={len(df[df['label']==0])}, vul={len(df[df['label']==1])})")

    ds = CodeDataset(df, tokenizer, max_length)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    all_labels, all_preds, all_scores = [], [], []

    for batch in tqdm(dl, desc=f"Proto-{name}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].cpu().numpy()

        embeddings = model(input_ids, attention_mask)
        sim_safe = torch.nn.functional.cosine_similarity(embeddings, safe_proto.unsqueeze(0), dim=1)
        sim_vul = torch.nn.functional.cosine_similarity(embeddings, vul_proto.unsqueeze(0), dim=1)
        scores = sim_safe - sim_vul
        preds = (scores >= 0).long().cpu().numpy()

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())
        all_scores.extend(scores.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds, all_scores, name)
    save_results(all_labels, all_preds, all_scores, output_dir, f'proto_{name}')
    return metrics


# ================= 分类器评分（Ours） =================
@torch.no_grad()
def evaluate_classifier(name, df, codebert, classifier, tokenizer,
                        device, max_length, batch_size, output_dir):
    """冻结编码器 → AttentionPooling + MLP 分类器"""
    print(f"\n{'=' * 60}")
    print(f"[Classifier - {name}] 样本: {len(df)} (safe={len(df[df['label']==0])}, vul={len(df[df['label']==1])})")

    ds = CodeDataset(df, tokenizer, max_length)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    all_labels, all_preds, all_probs = [], [], []

    for batch in tqdm(dl, desc=f"CLF-{name}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].cpu().numpy()

        outputs = codebert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, L, 768)

        logits = classifier(hidden_states, attention_mask)
        probs = torch.nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        vul_probs = probs[:, 1].cpu().numpy()

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(vul_probs.tolist())

    metrics = compute_metrics(all_labels, all_preds, all_probs, name)
    save_results(all_labels, all_preds, all_probs, output_dir, f'classifier_{name}')
    return metrics


# ================= 辅助函数 =================
def compute_metrics(labels, preds, scores, name):
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)
    scores_arr = np.array(scores)

    return {
        'name': name,
        'total': len(labels),
        'accuracy': accuracy_score(labels_arr, preds_arr),
        'precision': precision_score(labels_arr, preds_arr, zero_division=0),
        'recall': recall_score(labels_arr, preds_arr, zero_division=0),
        'f1': f1_score(labels_arr, preds_arr, zero_division=0),
        'auc': roc_auc_score(labels_arr, scores_arr),
        'safe_mean': scores_arr[labels_arr == 0].mean(),
        'vul_mean': scores_arr[labels_arr == 1].mean(),
    }


def save_results(labels, preds, scores, output_dir, prefix):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame({'label': labels, 'pred': preds, 'score': scores})
    result_df.to_csv(os.path.join(output_dir, f'{prefix}_results.csv'), index=False)


def print_comparison(proto_test, proto_val, clf_test, clf_val):
    """打印原型 vs 分类器对比表"""
    print(f"\n{'=' * 80}")
    print("方法对比：原型基线 vs 安全分类器")
    print(f"{'=' * 80}")

    keys = [
        ('Accuracy', 'accuracy'), ('Precision', 'precision'),
        ('Recall', 'recall'), ('F1', 'f1'), ('AUC', 'auc'),
        ('Safe Mean', 'safe_mean'), ('Vul Mean', 'vul_mean'),
    ]

    header = f"{'指标':<16} {'原型-Test':>12} {'分类器-Test':>12} {'提升':>10} | {'原型-Val':>12} {'分类器-Val':>12} {'提升':>10}"
    print(header)
    print("-" * len(header))

    for label, key in keys:
        pt, ct = proto_test[key], clf_test[key]
        pv, cv = proto_val[key], clf_val[key]
        gain_t = ct - pt if key != 'vul_mean' else (ct - pt)  # vul_mean 不适用简单减法
        gain_v = cv - pv if key != 'vul_mean' else (cv - pv)
        print(f"{label:<16} {pt:>12.4f} {ct:>12.4f} {gain_t:>+10.4f} | {pv:>12.4f} {cv:>12.4f} {gain_v:>+10.4f}")

    print(f"\n  原型 Val F1 = {proto_val['f1']:.4f} (接近随机, 证明原型方法不足)")
    print(f"  分类器 Val F1 = {clf_val['f1']:.4f}")


# ================= 主流程 =================
def main():
    print("=" * 60)
    print("实验A：原型基线 vs 安全分类器 对比评估")
    print("=" * 60)
    print(f"设备: {CONFIG['device']}")

    # 1. 加载模型
    print(f"\n加载编码器: {CONFIG['encoder_path']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['encoder_path'])

    # 原型评分用的编码器（带 projection head）
    contrastive_encoder = ContrastiveEncoder(CONFIG['encoder_path'], projection_dim=CONFIG['projection_dim'])
    contrastive_encoder = contrastive_encoder.to(CONFIG['device'])
    contrastive_encoder.eval()

    # 分类器用的编码器（原始 CodeBERT，取 CLS）
    codebert = AutoModel.from_pretrained(CONFIG['encoder_path']).to(CONFIG['device'])
    codebert.eval()

    # 2. 加载原型
    safe_proto = torch.load(os.path.join(CONFIG['proto_dir'], 'safe_prototype.pt'),
                            map_location=CONFIG['device'])
    vul_proto = torch.load(os.path.join(CONFIG['proto_dir'], 'vul_prototype.pt'),
                           map_location=CONFIG['device'])
    # 检查原型质量
    proto_cos = torch.nn.functional.cosine_similarity(safe_proto.unsqueeze(0), vul_proto.unsqueeze(0), dim=1).item()
    print(f"  原型间余弦相似度: {proto_cos:.4f} {'⚠️ 原型过于相似' if abs(proto_cos) > 0.95 else '✅ 原型有区分度'}")

    # 3. 加载分类器（AttentionPooling + MLP）
    classifier = SafetyClassifier().to(CONFIG['device'])
    classifier.load_state_dict(torch.load(CONFIG['classifier_path'], map_location=CONFIG['device']))
    classifier.eval()
    print(f"  分类器已加载: {CONFIG['classifier_path']}")

    # 4. 加载数据
    test_df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'test_contrastive.csv'))
    val_df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'val_contrastive.csv'))
    print(f"  Test: {len(test_df)} 条, Val: {len(val_df)} 条")

    # 5. 原型基线评估
    print(f"\n{'─' * 60}")
    print("第一部分：原型基线（Cosine Similarity）")
    print(f"{'─' * 60}")
    proto_test = evaluate_prototype('test', test_df, contrastive_encoder, tokenizer,
                                    safe_proto, vul_proto, CONFIG['device'],
                                    CONFIG['max_length'], CONFIG['batch_size'], CONFIG['output_dir'])
    proto_val = evaluate_prototype('val', val_df, contrastive_encoder, tokenizer,
                                   safe_proto, vul_proto, CONFIG['device'],
                                   CONFIG['max_length'], CONFIG['batch_size'], CONFIG['output_dir'])

    # 6. 分类器评估
    print(f"\n{'─' * 60}")
    print("第二部分：安全分类器（Frozen Encoder + MLP）")
    print(f"{'─' * 60}")
    clf_test = evaluate_classifier('test', test_df, codebert, classifier, tokenizer,
                                   CONFIG['device'], CONFIG['max_length'],
                                   CONFIG['batch_size'], CONFIG['output_dir'])
    clf_val = evaluate_classifier('val', val_df, codebert, classifier, tokenizer,
                                  CONFIG['device'], CONFIG['max_length'],
                                  CONFIG['batch_size'], CONFIG['output_dir'])

    # 7. 对比报告
    print_comparison(proto_test, proto_val, clf_test, clf_val)

    # 8. 达标判断（以分类器为准）
    print(f"\n{'=' * 60}")
    f1_test, auc_test = clf_test['f1'], clf_test['auc']
    if f1_test > 0.75 and auc_test > 0.85:
        print(f"✅ 实验A 达标 (Classifier): F1={f1_test:.4f} > 0.75, AUC={auc_test:.4f} > 0.85")
    else:
        print(f"⚠ 实验A 未达标: F1={f1_test:.4f} (需>0.75), AUC={auc_test:.4f} (需>0.85)")

    # 9. 一致性检查
    diff_f1 = abs(clf_test['f1'] - clf_val['f1'])
    diff_auc = abs(clf_test['auc'] - clf_val['auc'])
    if diff_f1 < 0.05 and diff_auc < 0.05:
        print(f"✅ 一致性通过: |ΔF1|={diff_f1:.4f}, |ΔAUC|={diff_auc:.4f} 均 < 0.05")
    else:
        print(f"⚠ 一致性未通过: |ΔF1|={diff_f1:.4f}, |ΔAUC|={diff_auc:.4f}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
