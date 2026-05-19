#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
t-SNE 可视化：查看漏洞/修复代码嵌入的可分性
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# ============ 配置 ============
CONFIG = {
    'model_path': '/root/autodl-tmp/outputs/models/best_model',
    'data_dir': '/root/autodl-tmp/SecurePath/SecureProtoCoT/data/processed',
    'max_length': 512,
    'sample_size': 400,  # 可视化样本数（每类 200）
    'random_seed': 42,
    'output_path': '/root/autodl-tmp/outputs/visualizations/tsne_plot.png',
}

# 设置随机种子
np.random.seed(CONFIG['random_seed'])
torch.manual_seed(CONFIG['random_seed'])

def load_model_and_tokenizer(model_path):
    """加载训练好的模型和 tokenizer"""
    print(f"📦 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoder = AutoModel.from_pretrained(model_path)
    encoder.eval()
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        print("✅ 使用 GPU")
    return encoder, tokenizer

def encode_codes(model, tokenizer, codes, device, max_length=512):
    """批量编码代码为嵌入向量"""
    embeddings = []
    batch_size = 16
    
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        encodings = tokenizer(
            batch,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**encodings)
            cls_embeds = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            embeddings.append(cls_embeds.cpu().numpy())
    
    return np.vstack(embeddings)

def load_and_sample_data(data_dir, sample_size=400):
    """加载测试数据并采样"""
    print("📊 加载测试数据...")
    test_df = pd.read_csv(os.path.join(data_dir, 'test_ratio.csv'))
    
    # 分离漏洞和修复样本
    vul_samples = []
    fix_samples = []
    
    for idx, row in test_df.iterrows():
        code = str(row['func_before']) if idx % 2 == 0 else str(row['func_after'])
        label = 1 if idx % 2 == 0 else 0  # 1=漏洞, 0=修复
        
        if label == 1 and len(vul_samples) < sample_size // 2:
            vul_samples.append(code)
        elif label == 0 and len(fix_samples) < sample_size // 2:
            fix_samples.append(code)
        
        if len(vul_samples) + len(fix_samples) >= sample_size:
            break
    
    print(f"✅ 采样完成: 漏洞={len(vul_samples)}, 修复={len(fix_samples)}")
    return vul_samples, fix_samples

def plot_tsne(embeddings, labels, output_path):
    """t-SNE 降维并绘图"""
    print("🎨 执行 t-SNE 降维...")
    
    # t-SNE 参数
    tsne = TSNE(
        n_components=2,
        perplexity=30,      # 可根据样本数调整 (5-50)
        learning_rate='auto',
        init='pca',
        random_state=CONFIG['random_seed'],
        n_iter=1000,
        verbose=1
    )
    
    # 降维
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    
    # 漏洞样本 (红色)
    vul_mask = labels == 1
    plt.scatter(
        embeddings_2d[vul_mask, 0], 
        embeddings_2d[vul_mask, 1],
        c='#FF6B6B', 
        label='Vulnerable Code',
        alpha=0.6,
        s=30,
        edgecolors='white',
        linewidth=0.5
    )
    
    # 修复样本 (蓝色)
    fix_mask = labels == 0
    plt.scatter(
        embeddings_2d[fix_mask, 0], 
        embeddings_2d[fix_mask, 1],
        c='#4ECDC4', 
        label='Fixed Code',
        alpha=0.6,
        s=30,
        edgecolors='white',
        linewidth=0.5
    )
    
    plt.title('t-SNE Visualization: Vulnerable vs Fixed Code Embeddings', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(fontsize=10, frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 图片已保存: {output_path}")
    
    # 显示（如果有 GUI 环境）
    # plt.show()
    plt.close()
    
    return embeddings_2d

def calculate_separation_score(embeddings_2d, labels):
    """简单计算两类样本的分离程度"""
    vul_embeds = embeddings_2d[labels == 1]
    fix_embeds = embeddings_2d[labels == 0]
    
    # 计算类内距离和类间距离
    vul_center = vul_embeds.mean(axis=0)
    fix_center = fix_embeds.mean(axis=0)
    
    within_vul = np.mean([np.linalg.norm(e - vul_center) for e in vul_embeds])
    within_fix = np.mean([np.linalg.norm(e - fix_center) for e in fix_embeds])
    between = np.linalg.norm(vul_center - fix_center)
    
    # 分离分数: 类间距离 / 平均类内距离 (越大越好)
    separation = between / ((within_vul + within_fix) / 2)
    
    print(f"\n📐 分离度分析:")
    print(f"   漏洞类中心: ({vul_center[0]:.3f}, {vul_center[1]:.3f})")
    print(f"   修复类中心: ({fix_center[0]:.3f}, {fix_center[1]:.3f})")
    print(f"   类间距离: {between:.3f}")
    print(f"   平均类内距离: {(within_vul + within_fix)/2:.3f}")
    print(f"   🔍 分离分数: {separation:.3f} (≥1.5 表示较好分离)")
    
    return separation

def main():
    print("=" * 60)
    print("🔍 t-SNE 嵌入可视化分析")
    print("=" * 60)
    
    # 1. 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_model_and_tokenizer(CONFIG['model_path'])
    
    # 2. 加载并采样数据
    vul_codes, fix_codes = load_and_sample_data(
        CONFIG['data_dir'], 
        sample_size=CONFIG['sample_size']
    )
    
    # 3. 编码为嵌入
    print("🔄 编码嵌入向量...")
    vul_embeds = encode_codes(model, tokenizer, vul_codes, device, CONFIG['max_length'])
    fix_embeds = encode_codes(model, tokenizer, fix_codes, device, CONFIG['max_length'])
    
    # 合并
    all_embeds = np.vstack([vul_embeds, fix_embeds])
    all_labels = np.array([1] * len(vul_embeds) + [0] * len(fix_embeds))
    
    print(f"✅ 嵌入维度: {all_embeds.shape}")
    
    # 4. t-SNE 可视化
    embeddings_2d = plot_tsne(all_embeds, all_labels, CONFIG['output_path'])
    
    # 5. 计算分离分数
    separation = calculate_separation_score(embeddings_2d, all_labels)
    
    # 6. 结论建议
    print(f"\n💡 分析结论:")
    if separation >= 2.0:
        print("   🟢 特征分离良好！问题可能在评估方式（原型距离）")
        print("   → 建议：尝试线性分类头替代原型距离")
    elif separation >= 1.5:
        print("   🟡 特征有一定分离，但有重叠")
        print("   → 建议：增加训练数据 / 调整对比学习温度系数")
    else:
        print("   🔴 特征几乎不可分，模型未学到有效区分信号")
        print("   → 建议：检查 Loss 逻辑 / 数据配对 / 增加训练轮数")
    
    print(f"\n📁 可视化图片: {CONFIG['output_path']}")
    print("=" * 60)

if __name__ == '__main__':
    main()