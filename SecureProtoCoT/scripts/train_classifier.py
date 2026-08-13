#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
安全分类器训练脚本（Attention Pooling 版）
在冻结的 CodeBERT 之上，训练 AttentionPooling + MLP 二分类器
学到的注意力权重可揭示模型关注哪些 token 来判别安全性
"""
import os, sys, torch, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
from pathlib import Path

# 导入 attention pooling 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from attention_pooling import SafetyClassifier

# ================= 配置 =================
CONFIG = {
    'encoder_path': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/models/best_model',
    'data_dir': '/home2/zzl/SecurePath/SecureProtoCoT/data/processed',
    'output_dir': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/models',
    'max_length': 512,
    'batch_size': 32,           # hidden_states 更大，batch 适当减小
    'learning_rate': 1e-3,
    'num_epochs': 30,
    'patience': 5,
    'hidden_dim': 256,
    'dropout': 0.3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


# ================= 数据集 =================
class CodeDataset(Dataset):
    """存储原始代码字符串和标签"""
    def __init__(self, codes, labels):
        self.codes = codes
        self.labels = labels

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        return self.codes[idx], self.labels[idx]


# ================= 编码（on-the-fly，编码器冻结） =================
@torch.no_grad()
def encode_batch(codes, encoder, tokenizer, device, max_length):
    """返回 last_hidden_state (B, L, H) + attention_mask"""
    enc = tokenizer(codes, max_length=max_length, padding=True,
                    truncation=True, return_tensors='pt')
    outputs = encoder(input_ids=enc['input_ids'].to(device),
                      attention_mask=enc['attention_mask'].to(device))
    return outputs.last_hidden_state, enc['attention_mask'].to(device)


# ================= 评估 =================
@torch.no_grad()
def evaluate(classifier, dataset, encoder, tokenizer, device, max_length, batch_size=64):
    """评估 AttentionPooling 分类器"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels, all_probs = [], [], []

    classifier.eval()
    for codes, labels in loader:
        hidden_states, attn_mask = encode_batch(
            codes, encoder, tokenizer, device, max_length)
        logits = classifier(hidden_states, attn_mask)
        probs = torch.nn.functional.softmax(logits, dim=1)

        all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs),
    }


# ================= 主流程 =================
def main():
    print("=" * 60)
    print("安全分类器训练（Attention Pooling + MLP）")
    print("=" * 60)
    print(f"设备: {CONFIG['device']}")

    # 1. 加载冻结编码器
    print(f"\n加载编码器: {CONFIG['encoder_path']}")
    encoder = AutoModel.from_pretrained(CONFIG['encoder_path']).to(CONFIG['device'])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['encoder_path'])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print("编码器已冻结")

    # 2. 加载数据
    print("\n加载数据...")
    train_df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'train_contrastive.csv'))
    val_df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'val_contrastive.csv'))
    test_df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'test_contrastive.csv'))

    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        n_vul = (df['label'] == 1).sum()
        n_safe = (df['label'] == 0).sum()
        print(f"  {name}: {len(df)} 条 (vul={n_vul}, safe={n_safe})")

    # 构建 Dataset（存储原始代码）
    train_ds = CodeDataset(
        train_df['code'].astype(str).tolist(),
        train_df['label'].astype(int).tolist())
    val_ds = CodeDataset(
        val_df['code'].astype(str).tolist(),
        val_df['label'].astype(int).tolist())
    test_ds = CodeDataset(
        test_df['code'].astype(str).tolist(),
        test_df['label'].astype(int).tolist())

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)

    # 3. 创建分类器（AttentionPooling + MLP）
    classifier = SafetyClassifier(
        input_dim=768,
        hidden_dim=CONFIG['hidden_dim'],
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])

    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"\n分类器参数量: {total_params:,} (AttentionPooling + MLP)")

    optimizer = AdamW(classifier.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
    loss_fn = CrossEntropyLoss()

    # 4. 训练
    print(f"\n开始训练 ({CONFIG['num_epochs']} epochs, patience={CONFIG['patience']})")
    best_val_f1 = 0
    best_state = None
    patience_counter = 0

    for epoch in range(CONFIG['num_epochs']):
        classifier.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        for codes, labels in pbar:
        # 🔑 关键：先移好 labels，后续都用这个
            labels = labels.to(CONFIG['device'])
    
            hidden_states, attn_mask = encode_batch(
            codes, encoder, tokenizer, CONFIG['device'], CONFIG['max_length'])

            logits = classifier(hidden_states, attn_mask)
            loss = loss_fn(logits, labels)  # 直接用，不用重复 .to()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()  # ✅ 都在 cuda:0
            total += len(labels)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.3f}'})

        train_acc = correct / total

        # 验证
        val_metrics = evaluate(classifier, val_ds, encoder, tokenizer,
                              CONFIG['device'], CONFIG['max_length'])

        print(f"Epoch {epoch+1:2d} | Train Acc: {train_acc:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_state = {k: v.clone() for k, v in classifier.state_dict().items()}
            patience_counter = 0
            print(f"  ↑ 最佳模型 (Val F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # 5. 测试
    classifier.load_state_dict(best_state)
    test_metrics = evaluate(classifier, test_ds, encoder, tokenizer,
                           CONFIG['device'], CONFIG['max_length'])

    print(f"\n{'=' * 60}")
    print("最终评估 (Test Set)")
    print(f"{'=' * 60}")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 6. 保存
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_dir / 'safety_classifier_attn.pt')
    print(f"\n分类器已保存: {output_dir / 'safety_classifier_attn.pt'}")

    print(f"\n{'=' * 60}")
    if test_metrics['f1'] > 0.75 and test_metrics['auc'] > 0.85:
        print(f"✅ 达标: F1={test_metrics['f1']:.4f} > 0.75, AUC={test_metrics['auc']:.4f} > 0.85")
    else:
        print(f"⚠ 未达标: F1={test_metrics['f1']:.4f} (需>0.75), AUC={test_metrics['auc']:.4f} (需>0.85)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
