#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
留一法跨CWE泛化实验 (Leave-One-CWE-Out)
每次留出一种CWE作为测试集，其余CWE用于训练/验证，评估跨漏洞类型的泛化能力
"""
import os, sys, torch, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from attention_pooling import SafetyClassifier

# ================= 配置 =================
CONFIG = {
    'encoder_path': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/models/best_model',
    'data_dir': '/home2/zzl/SecurePath/SecureProtoCoT/data/processed',
    'output_dir': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/results',
    'max_length': 512,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'num_epochs': 20,
    'patience': 5,
    'hidden_dim': 256,
    'dropout': 0.3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# ================= 数据集 =================
class CodeDataset(Dataset):
    def __init__(self, codes, labels):
        self.codes = codes
        self.labels = labels
    def __len__(self):
        return len(self.codes)
    def __getitem__(self, idx):
        return self.codes[idx], self.labels[idx]

# ================= 编码 =================
@torch.no_grad()
def encode_batch(codes, encoder, tokenizer, device, max_length):
    enc = tokenizer(codes, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    outputs = encoder(input_ids=enc['input_ids'].to(device), attention_mask=enc['attention_mask'].to(device))
    return outputs.last_hidden_state, enc['attention_mask'].to(device)

# ================= 评估 =================
@torch.no_grad()
def evaluate(classifier, dataset, encoder, tokenizer, device, max_length, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    classifier.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    pbar = tqdm(loader, desc="Eval", leave=False)
    for codes, labels in pbar:
        hidden_states, attn_mask = encode_batch(codes, encoder, tokenizer, device, max_length)
        logits = classifier(hidden_states, attn_mask)
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())
        
    return {
        'acc': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan')
    }

# ================= 训练单分类器 =================
def train_one_classifier(train_df, val_df, encoder, tokenizer, device):
    train_ds = CodeDataset(train_df['code'].astype(str).tolist(), train_df['label'].astype(int).tolist())
    val_ds = CodeDataset(val_df['code'].astype(str).tolist(), val_df['label'].astype(int).tolist())
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, pin_memory=True)
    
    classifier = SafetyClassifier(
        input_dim=768, hidden_dim=CONFIG['hidden_dim'], dropout=CONFIG['dropout']
    ).to(device)
    
    optimizer = AdamW(classifier.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
    loss_fn = CrossEntropyLoss().to(device)
    
    best_val_f1 = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(CONFIG['num_epochs']):
        classifier.train()
        total_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Train", leave=False)
        
        for codes, labels in pbar:
            # 🔑 核心修复：统一将 labels 移到 GPU，避免 cpu/cuda 冲突
            labels = labels.to(device)
            
            hidden_states, attn_mask = encode_batch(codes, encoder, tokenizer, device, CONFIG['max_length'])
            logits = classifier(hidden_states, attn_mask)
            loss = loss_fn(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()  # ✅ 现在 preds 和 labels 同在 device
            total += len(labels)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.3f}'})
            
        train_acc = correct / total
        val_metrics = evaluate(classifier, val_ds, encoder, tokenizer, device, CONFIG['max_length'])
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_state = {k: v.clone() for k, v in classifier.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                break
                
    classifier.load_state_dict(best_state)
    return classifier

# ================= 主流程 =================
def main():
    print("=" * 60)
    print("留一法跨CWE泛化实验 (Leave-One-CWE-Out)")
    print("=" * 60)
    
    df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'train_contrastive.csv'))
    print(f"全量数据: {len(df)} 条\n")
    
    # 合并测试集到全量数据中参与训练（或按需求拆分，此处假设全量数据已含各CWE）
    test_df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'test_contrastive.csv'))
    all_df = pd.concat([df, test_df], ignore_index=True)
    
    cwe_list = sorted(all_df['cwe_type'].unique())
    results = []
    
    encoder = AutoModel.from_pretrained(CONFIG['encoder_path']).to(CONFIG['device'])
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['encoder_path'])
    
    for leave_cwe in cwe_list:
        print(f"\n{'='*60}")
        print(f"留出 CWE: {leave_cwe}")
        print(f"{'='*60}")
        
        test_mask = all_df['cwe_type'] == leave_cwe
        train_final_df = all_df[~test_mask].sample(frac=1, random_state=42).reset_index(drop=True)
        test_cwe_df = all_df[test_mask].reset_index(drop=True)
        
        # 从训练集中切分验证集 (10%)
        val_size = max(1, int(len(train_final_df) * 0.1))
        val_df_cwe = train_final_df.iloc[:val_size]
        train_df_cwe = train_final_df.iloc[val_size:]
        
        print(f"  训练: {len(train_df_cwe)} 条 (vul={train_df_cwe['label'].sum()}, safe={(train_df_cwe['label']==0).sum()})")
        print(f"  验证: {len(val_df_cwe)} 条")
        print(f"  测试: {len(test_cwe_df)} 条")
        
        classifier = train_one_classifier(train_df_cwe, val_df_cwe, encoder, tokenizer, CONFIG['device'])
        
        test_ds = CodeDataset(test_cwe_df['code'].astype(str).tolist(), test_cwe_df['label'].astype(int).tolist())
        test_metrics = evaluate(classifier, test_ds, encoder, tokenizer, CONFIG['device'], CONFIG['max_length'])
        
        print(f"  Test ({leave_cwe}): Acc={test_metrics['acc']:.4f} | F1={test_metrics['f1']:.4f} | AUC={test_metrics['auc']:.4f}")
        
        results.append({
            'Left_Out_CWE': leave_cwe,
            'N_Test': len(test_cwe_df),
            'Acc': test_metrics['acc'],
            'F1': test_metrics['f1'],
            'AUC': test_metrics['auc']
        })
        
    # 汇总
    res_df = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print(f"{'CWE':<15} {'N':>6} {'Acc':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 80)
    for _, row in res_df.iterrows():
        print(f"{row['Left_Out_CWE']:<15} {row['N_Test']:>6} {row['Acc']:>8.4f} {row['F1']:>8.4f} {row['AUC']:>8.4f}")
    print(f"{'MEAN':<15} {'':>6} {res_df['Acc'].mean():>8.4f} {res_df['F1'].mean():>8.4f} {res_df['AUC'].mean():>8.4f}")
    
    # 保存
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    res_df.to_csv(os.path.join(CONFIG['output_dir'], 'leave_one_cwe_out.csv'), index=False)
    print(f"\n✅ 结果已保存: {os.path.join(CONFIG['output_dir'], 'leave_one_cwe_out.csv')}")

if __name__ == '__main__':
    main()