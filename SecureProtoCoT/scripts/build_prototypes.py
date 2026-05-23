#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
原型构建脚本
使用训练好的编码器构建漏洞原型和安全原型
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# 配置
CONFIG = {
    # 模型路径
    'model_path': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/models/best_model',

    # 数据路径
    'data_dir': '/home2/zzl/SecurePath/SecureProtoCoT/data/processed',
    'train_file': 'train_contrastive.csv',

    # 输出路径
    'output_dir': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/models',

    # 模型参数
    'projection_dim': 256,
    'max_length': 512,

    # 其他
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


class ContrastiveEncoder(nn.Module):
    """对比学习编码器（需要与训练时一致）"""

    def __init__(self, model_path, projection_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
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


def load_model(model_path, output_dir, device):
    """加载模型"""
    print(f"加载模型: {model_path}")
    model = ContrastiveEncoder(model_path, projection_dim=CONFIG['projection_dim'])
    model = model.to(device)

    # 加载完整模型参数（编码器+投影头）
    state_dict_path = os.path.join(output_dir, 'model_state.pt')
    if os.path.exists(state_dict_path):
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
        print("已加载完整模型参数（含投影头）")
    else:
        print("⚠️ 未找到完整模型参数，投影头为随机初始化")

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"模型加载完成，设备: {device}")

    return model, tokenizer


def encode_codes(model, tokenizer, codes, device, max_length=512, batch_size=16):
    """批量编码代码"""
    embeddings = []

    for i in tqdm(range(0, len(codes), batch_size), desc="编码中"):
        batch = codes[i:i+batch_size]

        encodings = tokenizer(
            batch,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            embeds = model(
                encodings['input_ids'].to(device),
                encodings['attention_mask'].to(device)
            )
            embeddings.append(embeds.cpu())

    return torch.cat(embeddings, dim=0)


def build_prototypes(model, tokenizer, data_dir, train_file, device, max_length=512):
    """构建原型"""
    print("\n加载训练数据...")
    train_path = os.path.join(data_dir, train_file)
    df = pd.read_csv(train_path)

    print(f"总样本数: {len(df)}")
    print(f"  漏洞样本: {len(df[df['label'] == 1])}")
    print(f"  安全样本: {len(df[df['label'] == 0])}")

    # 分离漏洞和安全代码
    vul_codes = df[df['label'] == 1]['code'].astype(str).tolist()
    safe_codes = df[df['label'] == 0]['code'].astype(str).tolist()

    # 编码漏洞代码
    print(f"\n编码 {len(vul_codes)} 个漏洞样本...")
    vul_embeddings = encode_codes(model, tokenizer, vul_codes, device, max_length)

    # 编码安全代码
    print(f"编码 {len(safe_codes)} 个安全样本...")
    safe_embeddings = encode_codes(model, tokenizer, safe_codes, device, max_length)

    # 计算原型（均值向量）
    vul_prototype = vul_embeddings.mean(dim=0)
    safe_prototype = safe_embeddings.mean(dim=0)

    print(f"\n原型构建完成:")
    print(f"  漏洞原型: {vul_prototype.shape}")
    print(f"  安全原型: {safe_prototype.shape}")

    return vul_prototype, safe_prototype


def save_prototypes(vul_proto, safe_proto, output_dir):
    """保存原型"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vul_path = output_dir / 'vul_prototype.pt'
    safe_path = output_dir / 'safe_prototype.pt'

    torch.save(vul_proto, vul_path)
    torch.save(safe_proto, safe_path)

    print(f"\n原型已保存:")
    print(f"  {vul_path}")
    print(f"  {safe_path}")


def main():
    print("=" * 60)
    print("原型构建脚本")
    print("=" * 60)
    print(f"设备: {CONFIG['device']}")

    # 加载模型
    model, tokenizer = load_model(CONFIG['model_path'], CONFIG['output_dir'], CONFIG['device'])

    # 构建原型
    vul_proto, safe_proto = build_prototypes(
        model, tokenizer,
        CONFIG['data_dir'],
        CONFIG['train_file'],
        CONFIG['device'],
        CONFIG['max_length']
    )

    # 保存原型
    save_prototypes(vul_proto, safe_proto, CONFIG['output_dir'])

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
