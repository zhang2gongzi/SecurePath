"""
数据预处理脚本 - 方案二
功能：正确组织漏洞代码和安全代码池

方案二设计：
- 漏洞代码池: vul=1 的 func_before（真正的漏洞代码）
- 安全代码池: vul=0 的 func_before（安全代码） + vul=1 的 func_after（修复代码）

关键改进：漏洞和安全代码来自不同函数，差异明显
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from collections import Counter
import re

# 配置
CONFIG = {
    'input_path': r'E:\paper\new\database\MSR_data_cleaned\MSR_data_cleaned.csv',
    'output_dir': r'E:\paper\new\SecureProtoCoT\data\processed',

    # 6种目标CWE类型
    'target_cwes': [
        'CWE-119',  # 缓冲区溢出
        'CWE-416',  # Use After Free
        'CWE-125',  # 越界读取
        'CWE-476',  # 空指针解引用
        'CWE-190',  # 整数溢出
        'CWE-787',  # 越界写入
    ],

    # 采样配置
    'vul_samples_per_cwe': 100000,    # 用尽所有可用漏洞样本
    'safe_ratio': 1.0,               # 安全:漏洞 比例 (1.0 = 1:1 平衡)

    # 数据划分比例
    'train_ratio': 0.7,
    'val_ratio': 0.1,
    'test_ratio': 0.2,

    # 代码长度过滤
    'min_lines': 10,
    'max_lines': 200,

    # 随机种子
    'random_seed': 42,
}


def load_data(filepath):
    """加载数据"""
    print(f"正在加载数据: {filepath}")
    print("数据较大，请耐心等待...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"原始数据大小: {df.shape[0]} 行, {df.shape[1]} 列")
    return df


def filter_valid_samples(df):
    """过滤有效样本"""
    print("\n=== 过滤有效样本 ===")
    initial_count = len(df)

    # 1. func_before和func_after非空
    df = df[df['func_before'].notna() & df['func_after'].notna()].copy()
    print(f"过滤空函数后: {len(df)} (减少 {initial_count - len(df)})")

    # 2. 代码长度过滤
    def count_lines(code):
        if pd.isna(code) or not isinstance(code, str):
            return 0
        return len(code.split('\n'))

    df['func_before_lines'] = df['func_before'].apply(count_lines)
    df['func_after_lines'] = df['func_after'].apply(count_lines)

    df = df[
        (df['func_before_lines'] >= CONFIG['min_lines']) &
        (df['func_before_lines'] <= CONFIG['max_lines']) &
        (df['func_after_lines'] >= CONFIG['min_lines']) &
        (df['func_after_lines'] <= CONFIG['max_lines'])
    ].copy()
    print(f"长度过滤后: {len(df)} (减少 {initial_count - len(df)})")

    return df


def extract_cwe_number(cwe_str):
    """从CWE ID字符串中提取CWE编号"""
    if pd.isna(cwe_str):
        return None
    cwe_str = str(cwe_str).strip()
    match = re.search(r'CWE-(\d+)', cwe_str)
    if match:
        return f"CWE-{match.group(1)}"
    return None


def build_vul_and_safe_pools(df):
    """
    构建漏洞代码池和安全代码池

    漏洞代码池: vul=1 的 func_before
    安全代码池: vul=0 的 func_before + vul=1 的 func_after
    """
    print("\n=== 构建漏洞/安全代码池 ===")

    # 提取CWE编号
    df['cwe_type'] = df['CWE ID'].apply(extract_cwe_number)

    # 分离 vul=0 和 vul=1 的记录
    df_vul0 = df[df['vul'] == 0].copy()  # 安全代码记录
    df_vul1 = df[df['vul'] == 1].copy()  # 漏洞代码记录

    print(f"vul=0 记录数: {len(df_vul0)}")
    print(f"vul=1 记录数: {len(df_vul1)}")

    # 构建漏洞代码池：vul=1 的 func_before
    vul_pool = []
    for _, row in df_vul1.iterrows():
        vul_pool.append({
            'code': row['func_before'],
            'label': 1,  # 漏洞
            'cwe_type': row['cwe_type'],
            'project': row.get('project', 'unknown'),
            'source': 'vul_func_before'
        })

    # 构建安全代码池：vul=0 的 func_before + vul=1 的 func_after
    safe_pool = []

    # 来源1：vul=0 的 func_before（真正的安全代码）
    for _, row in df_vul0.iterrows():
        safe_pool.append({
            'code': row['func_before'],
            'label': 0,  # 安全
            'cwe_type': row['cwe_type'],
            'project': row.get('project', 'unknown'),
            'source': 'safe_func_before'
        })

    # 来源2：vul=1 的 func_after（修复后的代码）
    for _, row in df_vul1.iterrows():
        safe_pool.append({
            'code': row['func_after'],
            'label': 0,  # 安全
            'cwe_type': row['cwe_type'],
            'project': row.get('project', 'unknown'),
            'source': 'fixed_func_after'
        })

    vul_df = pd.DataFrame(vul_pool)
    safe_df = pd.DataFrame(safe_pool)

    print(f"\n漏洞代码池: {len(vul_df)} 条")
    print(f"安全代码池: {len(safe_df)} 条")
    print(f"  - 来自 vul=0 的安全代码: {len(safe_df[safe_df['source'] == 'safe_func_before'])}")
    print(f"  - 来自 vul=1 的修复代码: {len(safe_df[safe_df['source'] == 'fixed_func_after'])}")

    return vul_df, safe_df


def sample_balanced_data(vul_df, safe_df):
    """
    平衡采样：对每种CWE类型，采样相同数量的漏洞和安全样本
    """
    print("\n=== 平衡采样 ===")

    vul_samples = []
    safe_samples = []
    stats = {}

    for cwe in CONFIG['target_cwes']:
        # 漏洞样本
        cwe_vul = vul_df[vul_df['cwe_type'] == cwe]
        vul_available = len(cwe_vul)
        vul_n = min(vul_available, CONFIG['vul_samples_per_cwe'])

        if vul_n > 0:
            sampled_vul = cwe_vul.sample(n=vul_n, random_state=CONFIG['random_seed'])
            vul_samples.append(sampled_vul)

        # 安全样本：按比例匹配
        cwe_safe = safe_df[safe_df['cwe_type'] == cwe]
        safe_available = len(cwe_safe)
        safe_n = min(safe_available, max(1, int(vul_n * CONFIG['safe_ratio'])))

        if safe_n > 0:
            sampled_safe = cwe_safe.sample(n=safe_n, random_state=CONFIG['random_seed'])
            safe_samples.append(sampled_safe)

        stats[cwe] = {
            'vul_available': vul_available,
            'vul_sampled': vul_n,
            'safe_available': safe_available,
            'safe_sampled': safe_n
        }
        print(f"{cwe}: 漏洞 {vul_n}/{vul_available}, 安全 {safe_n}/{safe_available}")

    vul_sampled_df = pd.concat(vul_samples, ignore_index=True) if vul_samples else pd.DataFrame()
    safe_sampled_df = pd.concat(safe_samples, ignore_index=True) if safe_samples else pd.DataFrame()

    print(f"\n总采样: 漏洞 {len(vul_sampled_df)}, 安全 {len(safe_sampled_df)}")

    return vul_sampled_df, safe_sampled_df, stats


def split_data(vul_df, safe_df):
    """
    划分训练/验证/测试集
    确保漏洞和安全样本按比例划分
    """
    print("\n=== 划分数据集 ===")

    def split_single(df, name):
        n = len(df)
        df = df.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)

        train_end = int(n * CONFIG['train_ratio'])
        val_end = train_end + int(n * CONFIG['val_ratio'])

        train = df[:train_end]
        val = df[train_end:val_end]
        test = df[val_end:]

        print(f"{name}: 训练 {len(train)}, 验证 {len(val)}, 测试 {len(test)}")
        return train, val, test

    vul_train, vul_val, vul_test = split_single(vul_df, "漏洞样本")
    safe_train, safe_val, safe_test = split_single(safe_df, "安全样本")

    # 合并
    train_df = pd.concat([vul_train, safe_train], ignore_index=True)
    val_df = pd.concat([vul_val, safe_val], ignore_index=True)
    test_df = pd.concat([vul_test, safe_test], ignore_index=True)

    # 打乱
    train_df = train_df.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)

    print(f"\n总计: 训练 {len(train_df)}, 验证 {len(val_df)}, 测试 {len(test_df)}")
    print(f"训练集标签分布: 漏洞={len(train_df[train_df['label']==1])}, 安全={len(train_df[train_df['label']==0])}")

    return train_df, val_df, test_df


def save_data(df, filename, output_dir):
    """保存数据"""
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"已保存: {filepath}")
    return filepath


def main():
    print("=" * 60)
    print("MSR_data_cleaned 数据预处理 - 方案二")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    df = load_data(CONFIG['input_path'])

    # 2. 过滤有效样本
    df = filter_valid_samples(df)

    # 3. 构建漏洞/安全代码池
    vul_df, safe_df = build_vul_and_safe_pools(df)

    # 4. 平衡采样
    vul_sampled, safe_sampled, sampling_stats = sample_balanced_data(vul_df, safe_df)

    # 5. 划分数据集
    train_df, val_df, test_df = split_data(vul_sampled, safe_sampled)

    # 6. 保存数据
    print("\n=== 保存数据 ===")
    save_data(train_df, 'train_contrastive.csv', output_dir)
    save_data(val_df, 'val_contrastive.csv', output_dir)
    save_data(test_df, 'test_contrastive.csv', output_dir)

    # 保存统计信息
    stats = {
        'sampling': sampling_stats,
        'final': {
            'train': {
                'total': len(train_df),
                'vul': len(train_df[train_df['label']==1]),
                'safe': len(train_df[train_df['label']==0])
            },
            'val': {
                'total': len(val_df),
                'vul': len(val_df[val_df['label']==1]),
                'safe': len(val_df[val_df['label']==0])
            },
            'test': {
                'total': len(test_df),
                'vul': len(test_df[test_df['label']==1]),
                'safe': len(test_df[test_df['label']==0])
            }
        }
    }

    stats_path = os.path.join(output_dir, 'cwe_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"已保存统计信息: {stats_path}")

    print("\n" + "=" * 60)
    print("数据预处理完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
