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
    'model_name': r'E:\paper\new\model\codebert-base',  # 本地模型路径
    'max_length': 512,
    'hidden_size': 768,
    'projection_dim': 256,

    # 训练
    'batch_size': 8,
    'learning_rate': 2e-5,
    'num_epochs': 5,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,

    # 对比学习
    'temperature': 0.07,

    # 路径
    'data_dir': r'E:\paper\new\SecureProtoCoT\data\processed',
    'output_dir': r'E:\paper\new\SecureProtoCoT\outputs\models',

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
    """对比学习损失函数"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, vul_embeds, fix_embeds):
        """
        Args:
            vul_embeds: 漏洞代码嵌入 [batch_size, projection_dim]
            fix_embeds: 修复代码嵌入 [batch_size, projection_dim]
        """
        batch_size = vul_embeds.size(0)

        # 计算相似度矩阵
        similarity = torch.mm(vul_embeds, fix_embeds.t()) / self.temperature

        # 对角线是正样本对（同一函数的漏洞版本和修复版本）
        # 我们希望：漏洞代码远离修复代码
        # 所以正样本是同类漏洞，负样本是修复代码

        # 漏洞代码应该与其他漏洞代码相似
        vul_sim = torch.mm(vul_embeds, vul_embeds.t()) / self.temperature

        # 修复代码应该与其他修复代码相似
        fix_sim = torch.mm(fix_embeds, fix_embeds.t()) / self.temperature

        # 损失：同类聚集，异类分离
        # 简化版：使用infoNCE
        labels = torch.arange(batch_size).to(vul_embeds.device)

        # 漏洞代码的损失：与其他漏洞相似，与修复代码不相似
        vul_loss = F.cross_entropy(similarity, labels)

        # 修复代码的损失：与其他修复相似，与漏洞代码不相似
        fix_loss = F.cross_entropy(similarity.t(), labels)

        loss = (vul_loss + fix_loss) / 2

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
