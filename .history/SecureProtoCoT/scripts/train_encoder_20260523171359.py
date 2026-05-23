"""
编码器训练脚本 - 方案二
使用对比学习训练CodeBERT，区分漏洞代码和安全代码

数据格式适配：
- 每行有 code 和 label 字段
- label=1 表示漏洞代码，label=0 表示安全代码
- 漏洞和安全代码来自不同函数（无配对关系）
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path

# 配置
CONFIG = {
    # 模型
    'model_name': r'/root/autodl-tmp/codebert-base',
    'max_length': 512,
    'hidden_size': 768,
    'projection_dim': 256,

    # 训练
    'batch_size': 16,
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
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # 投影
        projection = self.projection(cls_embedding)

        # L2归一化
        projection = F.normalize(projection, p=2, dim=1)

        return projection


class ContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Args:
        embeddings: [batch_size, projection_dim]
            labels: [batch_size], 1=漏洞, 0=安全
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        # 计算相似度矩阵
        similarity = torch.mm(embeddings, embeddings.t()) / self.temperature

        # 创建标签mask：同label为正样本对
        labels = labels.view(-1, 1)
        mask = (labels == labels.t()).float()

        # 排除自身
        mask.fill_diagonal_(0.0)

        # 数值稳定的 InfoNCE Loss
        sim_max = torch.max(similarity, dim=1, keepdim=True).values
        exp_sim = torch.exp(similarity - sim_max)

        # 分子：同类相似度之和
        pos_sum = (exp_sim * mask).sum(dim=1) + 1e-8
        # 分母：所有相似度之和（排除自身）
        all_sum = exp_sim.sum(dim=1) - torch.diag(exp_sim) + 1e-8

        loss = -torch.log(pos_sum / all_sum).mean()

        return loss


class CodeDataset(Dataset):
    """代码数据集"""

    def __init__(self, df, tokenizer, max_length):
        self.codes = df['code'].astype(str).tolist()
        self.labels = df['label'].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.codes[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx])
        }


def load_data(data_dir, tokenizer, max_length, batch_size):
    """加载数据"""
    print("加载数据...")

    # 加载训练数据
    df = pd.read_csv(os.path.join(data_dir, 'train_contrastive.csv'))
    print(f"训练数据: {len(df)} 条")
    print(f"  漏洞样本: {len(df[df['label'] == 1])}")
    print(f"  安全样本: {len(df[df['label'] == 0])}")

    # 创建数据集
    dataset = CodeDataset(df, tokenizer, max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    return dataloader


def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 前向传播
        embeddings = model(input_ids, attention_mask)

        # 计算损失
        loss = loss_fn(embeddings, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def evaluate(model, data_dir, tokenizer, device, max_length=512):
    """评估模型：计算漏洞检测准确率"""
    model.eval()

    # 加载测试数据
    test_df = pd.read_csv(os.path.join(data_dir, 'test_contrastive.csv'))
    print(f"\n测试数据: {len(test_df)} 条")

    # 加载训练数据构建原型
    train_df = pd.read_csv(os.path.join(data_dir, 'train_contrastive.csv'))

    vul_embeds_list = []
    safe_embeds_list = []

    print("构建漏洞原型和安全原型...")

    with torch.no_grad():
        # 采样漏洞代码构建漏洞原型
        vul_samples = train_df[train_df['label'] == 1].sample(n=min(100, len(train_df[train_df['label'] == 1])), random_state=42)
        for _, row in vul_samples.iterrows():
            code = str(row['code'])
            encoding = tokenizer(code, max_length=max_length,
                                padding=True, truncation=True, return_tensors='pt')
            embed = model(encoding['input_ids'].to(device),
                         encoding['attention_mask'].to(device))
            vul_embeds_list.append(embed.cpu())

        # 采样安全代码构建安全原型
        safe_samples = train_df[train_df['label'] == 0].sample(n=min(100, len(train_df[train_df['label'] == 0])), random_state=42)
        for _, row in safe_samples.iterrows():
            code = str(row['code'])
            encoding = tokenizer(code, max_length=max_length,
                                padding=True, truncation=True, return_tensors='pt')
            embed = model(encoding['input_ids'].to(device),
                         encoding['attention_mask'].to(device))
            safe_embeds_list.append(embed.cpu())

        # 计算原型
        vul_prototype = torch.cat(vul_embeds_list, dim=0).mean(dim=0)
        safe_prototype = torch.cat(safe_embeds_list, dim=0).mean(dim=0)

        print(f"原型构建完成: 漏洞原型 {vul_prototype.shape}, 安全原型 {safe_prototype.shape}")

    # 测试
    print("评估测试集...")
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, row in test_df.iterrows():
            code = str(row['code'])
            true_label = int(row['label'])  # 1=漏洞, 0=安全

            encoding = tokenizer(code, max_length=max_length,
                               padding=True, truncation=True, return_tensors='pt')
            embed = model(encoding['input_ids'].to(device),
                         encoding['attention_mask'].to(device))

            # 计算与原型的距离
            vul_dist = torch.norm(embed.cpu() - vul_prototype, p=2)
            safe_dist = torch.norm(embed.cpu() - safe_prototype, p=2)

            # 预测：离哪个原型更近
            pred_label = 1 if vul_dist < safe_dist else 0

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
    print("安全感知编码器训练 - 方案二")
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
            torch.save(model.state_dict(), output_dir / 'model_state.pt')
            tokenizer.save_pretrained(model_path)
            print(f"保存最佳模型到: {model_path}")

    print("\n" + "=" * 60)
    print(f"训练完成! 最佳准确率: {best_accuracy:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
