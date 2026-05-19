"""
编码器训练脚本
使用对比学习训练CodeBERT，区分漏洞代码和安全代码
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path

# 配置
CONFIG = {
    # 模型
    'model_name': r'/root/autodl-tmp/codebert-base',  # 本地模型路径
    'max_length': 512,
    'hidden_size': 768,
    'projection_dim': 256,

    # 训练
    'batch_size': 8,
    'gradient_accumulation_steps': 4,
    'learning_rate': 2e-5,
    'num_epochs': 10,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,

    # 对比学习
    'temperature': 0.07,

    # 路径
    'data_dir': r'/root/autodl-tmp/SecurePath/SecureProtoCoT/data/processed',
    'output_dir': r'/root/autodl-tmp/outputs/models',

    # 其他
    'random_seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


class ContrastiveEncoder(nn.Module):
    """对比学习编码器"""

    def __init__(self, model_name, projection_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        # 获取[CLS]表示
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # 投影
        projection = self.projection(cls_embedding)  # [batch_size, projection_dim]

        # L2归一化
        projection = F.normalize(projection, p=2, dim=1)

        return projection

class ContrastiveLoss(nn.Module):
    """对比学习损失函数：同类聚集，异类分离"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, vul_embeds, fix_embeds):
        """
        修复版：同类聚集，异类分离
        - vul[*] 之间应该相似（正样本）
        - fix[*] 之间应该相似（正样本）  
        - vul[*] 与 fix[*] 应该不相似（负样本）
        """
        batch_size = vul_embeds.size(0)
        device = vul_embeds.device
        
        # 🔍 调试打印（保留监控）
        with torch.no_grad():
            sim_matrix = torch.mm(vul_embeds, fix_embeds.t()) / self.temperature
            diag_sim = torch.diag(sim_matrix).mean().item()
            if batch_size > 1:
                offdiag_sum = sim_matrix.sum() - torch.diag(sim_matrix).sum()
                offdiag_sim = offdiag_sum / (batch_size**2 - batch_size)
            else:
                offdiag_sim = 0.0
            print(f"[DEBUG] diag={diag_sim:.4f}, offdiag={offdiag_sim:.4f}, diff={diag_sim-offdiag_sim:.4f}")

        # ✅ 核心修复：拼接所有嵌入，计算同类/异类相似度
        all_embeds = torch.cat([vul_embeds, fix_embeds], dim=0)  # [2B, D]
        
        # 计算全局相似度矩阵 [2B, 2B]
        similarity = torch.mm(all_embeds, all_embeds.t()) / self.temperature
        
        # 构造类别标签：前 B 个是 vul(0)，后 B 个是 fix(1)
        class_ids = torch.cat([
            torch.zeros(batch_size, device=device),
            torch.ones(batch_size, device=device)
        ])
        
        # 创建 mask：同类为 1，异类为 0，排除自身
        mask = (class_ids.unsqueeze(0) == class_ids.unsqueeze(1)).float()
        mask.fill_diagonal_(0.0)  # 排除自身对比
        
        # InfoNCE Loss: 最大化与同类样本的相似度（数值稳定版）
        sim_max = torch.max(similarity, dim=1, keepdim=True).values
        exp_sim = torch.exp(similarity - sim_max)
        
        pos_sum = (exp_sim * mask).sum(dim=1) + 1e-8  # 分子：同类相似度之和
        all_sum = exp_sim.sum(dim=1) + 1e-8            # 分母：所有相似度之和
        
        loss = -torch.log(pos_sum / all_sum).mean()
        return loss

def load_data(data_dir, tokenizer, max_length, batch_size):
    """加载数据"""
    print("加载数据...")

    # 加载对比学习数据
    df = pd.read_csv(os.path.join(data_dir, 'train_contrastive.csv'))
    print(f"训练数据: {len(df)} 条")

    # 配对数据
    paired_data = []
    for i in range(0, len(df), 2):
        if i + 1 < len(df):
            paired_data.append({
                'vulnerable': str(df.iloc[i]['code']),
                'fixed': str(df.iloc[i + 1]['code']),
            })

    print(f"配对样本: {len(paired_data)} 对")

    # 创建DataLoader
    def collate_fn(batch):
        vul_codes = [item['vulnerable'] for item in batch]
        fix_codes = [item['fixed'] for item in batch]

        vul_encodings = tokenizer(
            vul_codes,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        fix_encodings = tokenizer(
            fix_codes,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'vul_input_ids': vul_encodings['input_ids'],
            'vul_attention_mask': vul_encodings['attention_mask'],
            'fix_input_ids': fix_encodings['input_ids'],
            'fix_attention_mask': fix_encodings['attention_mask'],
        }

    dataloader = DataLoader(
        paired_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    return dataloader


def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        # 移动到设备
        vul_input_ids = batch['vul_input_ids'].to(device)
        vul_attention_mask = batch['vul_attention_mask'].to(device)
        fix_input_ids = batch['fix_input_ids'].to(device)
        fix_attention_mask = batch['fix_attention_mask'].to(device)

        # 前向传播
        vul_embeds = model(vul_input_ids, vul_attention_mask)
        fix_embeds = model(fix_input_ids, fix_attention_mask)

        # 计算损失
        loss = loss_fn(vul_embeds, fix_embeds)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def evaluate(model, data_dir, tokenizer, device, max_length=512):
    """评估模型：计算漏洞检测准确率"""
    model.eval()

    # 加载测试数据
    test_df = pd.read_csv(os.path.join(data_dir, 'test_ratio.csv'))
    print(f"\n测试数据: {len(test_df)} 条")

    correct = 0
    total = 0

    # 构建原型
    print("构建安全原型和漏洞原型...")

    train_df = pd.read_csv(os.path.join(data_dir, 'train_ratio.csv'))
    vul_embeds_list = []
    fix_embeds_list = []

    model.eval()
    with torch.no_grad():
        # 采样一部分构建原型
        sample_size = min(100, len(train_df) // 2)

        for i in range(0, sample_size * 2, 2):
            if i + 1 >= len(train_df):
                break

            # 漏洞代码
            vul_code = str(train_df.iloc[i]['func_before'])
            vul_encoding = tokenizer(vul_code, max_length=max_length,
                                    padding=True, truncation=True, return_tensors='pt')
            vul_embed = model(vul_encoding['input_ids'].to(device),
                             vul_encoding['attention_mask'].to(device))
            vul_embeds_list.append(vul_embed.cpu())

            # 修复代码
            fix_code = str(train_df.iloc[i]['func_after'])
            fix_encoding = tokenizer(fix_code, max_length=max_length,
                                    padding=True, truncation=True, return_tensors='pt')
            fix_embed = model(fix_encoding['input_ids'].to(device),
                             fix_encoding['attention_mask'].to(device))
            fix_embeds_list.append(fix_embed.cpu())

        # 计算原型
        vul_prototype = torch.cat(vul_embeds_list, dim=0).mean(dim=0)
        fix_prototype = torch.cat(fix_embeds_list, dim=0).mean(dim=0)

        print(f"原型构建完成: 漏洞原型 {vul_prototype.shape}, 安全原型 {fix_prototype.shape}")

    # 测试
    print("评估测试集...")
    with torch.no_grad():
        for idx, row in test_df.iterrows():
            code = str(row['func_before']) if idx % 2 == 0 else str(row['func_after'])
            true_label = 1 if idx % 2 == 0 else 0  # 1=漏洞, 0=安全

            encoding = tokenizer(code, max_length=max_length,
                               padding=True, truncation=True, return_tensors='pt')
            embed = model(encoding['input_ids'].to(device),
                         encoding['attention_mask'].to(device))

            # 计算与原型的距离
            vul_dist = torch.norm(embed.cpu() - vul_prototype, p=2)
            fix_dist = torch.norm(embed.cpu() - fix_prototype, p=2)

            # 预测：离哪个原型更近
            pred_label = 1 if vul_dist < fix_dist else 0

            if pred_label == true_label:
                correct += 1
            total += 1

            if total % 100 == 0:
                print(f"已处理 {total} 条, 当前准确率: {correct/total:.4f}")

    accuracy = correct / total
    print(f"\n测试准确率: {accuracy:.4f}")

    return accuracy


def main():
    print("=" * 60)
    print("安全感知编码器训练")
    print("=" * 60)
    print(f"设备: {CONFIG['device']}")

    # 设置随机种子
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])

    # 创建输出目录
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载tokenizer
    print(f"\n加载模型: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

    # 创建模型
    model = ContrastiveEncoder(
        CONFIG['model_name'],
        projection_dim=CONFIG['projection_dim']
    ).to(CONFIG['device'])

    # 加载数据
    dataloader = load_data(
        CONFIG['data_dir'],
        tokenizer,
        CONFIG['max_length'],
        CONFIG['batch_size']
    )

    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # 学习率调度
    num_training_steps = len(dataloader) * CONFIG['num_epochs']
    num_warmup_steps = int(num_training_steps * CONFIG['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 损失函数
    loss_fn = ContrastiveLoss(temperature=CONFIG['temperature'])

    # 训练
    print(f"\n开始训练，共 {CONFIG['num_epochs']} 个epoch")
    best_accuracy = 0

    for epoch in range(CONFIG['num_epochs']):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['num_epochs']} ---")

        avg_loss = train_epoch(
            model, dataloader, optimizer, scheduler, loss_fn, CONFIG['device']
        )
        print(f"平均损失: {avg_loss:.4f}")

        # 评估
        accuracy = evaluate(model, CONFIG['data_dir'], tokenizer, CONFIG['device'], CONFIG['max_length'])

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_path = output_dir / 'best_model'
            model.encoder.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            print(f"保存最佳模型到: {model_path}")

    print("\n" + "=" * 60)
    print(f"训练完成! 最佳准确率: {best_accuracy:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
