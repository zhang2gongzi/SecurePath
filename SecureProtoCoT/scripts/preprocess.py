"""
数据预处理脚本
功能：从MSR_data_cleaned采样并划分数据集
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from collections import Counter

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
    'samples_per_cwe': 1000,  # 每种CWE采样数量

    # 数据划分比例（策略一）
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

    # 使用low_memory=False避免类型警告
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

    # 尝试匹配 CWE-XXX 格式
    import re
    match = re.search(r'CWE-(\d+)', cwe_str)
    if match:
        return f"CWE-{match.group(1)}"

    return None


def sample_by_cwe(df):
    """按CWE类型采样"""
    print("\n=== 按CWE类型采样 ===")

    # 提取CWE编号
    df['cwe_type'] = df['CWE ID'].apply(extract_cwe_number)

    sampled_dfs = []
    stats = {}

    for cwe in CONFIG['target_cwes']:
        cwe_df = df[df['cwe_type'] == cwe].copy()
        available = len(cwe_df)

        if available >= CONFIG['samples_per_cwe']:
            sampled = cwe_df.sample(n=CONFIG['samples_per_cwe'], random_state=CONFIG['random_seed'])
        else:
            print(f"警告: {cwe} 只有 {available} 条样本，少于目标 {CONFIG['samples_per_cwe']}")
            sampled = cwe_df

        sampled_dfs.append(sampled)
        stats[cwe] = {
            'available': available,
            'sampled': len(sampled)
        }
        print(f"{cwe}: 可用 {available}, 采样 {len(sampled)}")

    result_df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"\n总采样数: {len(result_df)}")

    return result_df, stats


def split_strategy_one(df):
    """策略一：按比例划分"""
    print("\n=== 策略一：按比例划分 ===")

    train_list, val_list, test_list = [], [], []

    for cwe in CONFIG['target_cwes']:
        cwe_df = df[df['cwe_type'] == cwe].copy()
        n = len(cwe_df)

        # 打乱顺序
        cwe_df = cwe_df.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)

        # 计算划分点
        train_end = int(n * CONFIG['train_ratio'])
        val_end = train_end + int(n * CONFIG['val_ratio'])

        train_list.append(cwe_df[:train_end])
        val_list.append(cwe_df[train_end:val_end])
        test_list.append(cwe_df[val_end:])

        print(f"{cwe}: 训练 {train_end}, 验证 {val_end - train_end}, 测试 {n - val_end}")

    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    print(f"\n总计: 训练 {len(train_df)}, 验证 {len(val_df)}, 测试 {len(test_df)}")

    return train_df, val_df, test_df


def split_strategy_two(df, left_out_cwe='CWE-787'):
    """策略二：留一法划分（跨类型泛化实验）"""
    print(f"\n=== 策略二：留一法（留出 {left_out_cwe}）===")

    train_df = df[df['cwe_type'] != left_out_cwe].copy()
    test_df = df[df['cwe_type'] == left_out_cwe].copy()

    # 打乱顺序
    train_df = train_df.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)

    print(f"训练集 ({len(train_df)} 条):")
    for cwe in CONFIG['target_cwes']:
        if cwe != left_out_cwe:
            print(f"  {cwe}: {len(train_df[train_df['cwe_type'] == cwe])}")
    print(f"测试集 ({len(test_df)} 条): {left_out_cwe}")

    return train_df, test_df


def prepare_contrastive_data(df):
    """准备对比学习数据：漏洞代码(正样本) vs 修复代码(负样本)"""
    print("\n=== 准备对比学习数据 ===")

    samples = []

    for _, row in df.iterrows():
        # 漏洞代码作为正样本（label=1）
        samples.append({
            'code': row['func_before'],
            'label': 1,  # 漏洞
            'cwe_type': row['cwe_type'],
            'project': row.get('project', 'unknown'),
            'type': 'vulnerable'
        })

        # 修复代码作为负样本（label=0）
        samples.append({
            'code': row['func_after'],
            'label': 0,  # 安全
            'cwe_type': row['cwe_type'],
            'project': row.get('project', 'unknown'),
            'type': 'fixed'
        })

    result_df = pd.DataFrame(samples)
    print(f"对比学习样本数: {len(result_df)}")
    print(f"  漏洞样本: {len(result_df[result_df['label'] == 1])}")
    print(f"  安全样本: {len(result_df[result_df['label'] == 0])}")

    return result_df


def save_data(df, filename, output_dir):
    """保存数据"""
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"已保存: {filepath}")
    return filepath


def save_stats(stats, output_dir):
    """保存统计信息"""
    filepath = os.path.join(output_dir, 'cwe_stats.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"已保存统计信息: {filepath}")


def main():
    print("=" * 60)
    print("MSR_data_cleaned 数据预处理")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    df = load_data(CONFIG['input_path'])

    # 2. 过滤有效样本
    df = filter_valid_samples(df)

    # 3. 按CWE类型采样
    df_sampled, sampling_stats = sample_by_cwe(df)

    # 4. 策略一：按比例划分
    train_df, val_df, test_df = split_strategy_one(df_sampled)

    # 5. 策略二：留一法划分
    train_lo_df, test_lo_df = split_strategy_two(df_sampled, left_out_cwe='CWE-787')

    # 6. 准备对比学习数据
    train_contrastive = prepare_contrastive_data(train_df)

    # 7. 保存数据
    print("\n=== 保存数据 ===")

    # 策略一数据
    save_data(train_df, 'train_ratio.csv', output_dir)
    save_data(val_df, 'val_ratio.csv', output_dir)
    save_data(test_df, 'test_ratio.csv', output_dir)

    # 策略二数据
    save_data(train_lo_df, 'train_leave_one.csv', output_dir)
    save_data(test_lo_df, 'test_leave_one.csv', output_dir)

    # 对比学习数据
    save_data(train_contrastive, 'train_contrastive.csv', output_dir)

    # 保存统计信息
    stats = {
        'sampling': sampling_stats,
        'strategy_one': {
            'train': len(train_df),
            'val': len(val_df),
            'test': len(test_df)
        },
        'strategy_two': {
            'train': len(train_lo_df),
            'test': len(test_lo_df),
            'left_out_cwe': 'CWE-787'
        }
    }
    save_stats(stats, output_dir)

    print("\n" + "=" * 60)
    print("数据预处理完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
