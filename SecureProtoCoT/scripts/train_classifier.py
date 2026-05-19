#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
二分类训练脚本：直接优化 漏洞 vs 安全 判别任务
解决对比学习特征坍塌问题
"""
import os, sys, torch, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from pathlib import Path

# ================= 配置 =================
CONFIG = {
    'model_name': '/root/autodl-tmp/codebert-base',
    'data_dir': '/root/autodl-tmp/SecurePath/SecureProtoCoT/data/processed',
    'output_dir': '/root/autodl-tmp/outputs/models/best_classifier',
    'max_length': 512,
    'batch_size': 16,          # 二分类可跑更大 batch
    'learning_rate': 2e-5,
    'num_epochs': 5,
    'warmup_ratio': 0.1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ================= 数据集 =================
class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.codes = df['code'].astype(str).tolist()
        self.labels = df['label'].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.codes)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.codes[idx], max_length=self.max_length, padding='max_length', truncation=True)
        return {
            'input_ids': torch.tensor(enc['input_ids']),
            'attention_mask': torch.tensor(enc['attention_mask']),
            'labels': torch.tensor(self.labels[idx])
        }

# ================= 模型 =================
class VulnClassifier(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, 2)
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

# ================= 训练 =================
def train():
    print("="*50 + "\n🚀 二分类训练启动\n" + "="*50)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = VulnClassifier(CONFIG['model_name']).to(CONFIG['device'])
    
    # 数据加载
    train_df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'train_contrastive.csv'))
    train_ds = CodeDataset(train_df, tokenizer, CONFIG['max_length'])
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_dl)*CONFIG['num_epochs']*CONFIG['warmup_ratio']), num_training_steps=len(train_dl)*CONFIG['num_epochs'])
    loss_fn = CrossEntropyLoss()
    
    best_acc = 0
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        total_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        
        for batch in pbar:
            ids = batch['input_ids'].to(CONFIG['device'])
            mask = batch['attention_mask'].to(CONFIG['device'])
            labels = batch['labels'].to(CONFIG['device'])
            
            logits = model(ids, mask)
            loss = loss_fn(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 防梯度爆炸
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
            
        train_acc = correct / total
        print(f"✅ Epoch {epoch+1} 完成 | Train Loss: {total_loss/len(train_dl):.4f} | Train Acc: {train_acc:.4f}")
        
        # 保存最佳
        if train_acc > best_acc:
            best_acc = train_acc
            model.encoder.save_pretrained(CONFIG['output_dir'])
            tokenizer.save_pretrained(CONFIG['output_dir'])
            print(f"💾 保存最佳模型 (Acc: {best_acc:.4f})")
            
    print(f"\n🏆 训练完成! 最佳训练准确率: {best_acc:.4f}")
    print(f"📁 模型路径: {CONFIG['output_dir']}")

if __name__ == '__main__':
    train()