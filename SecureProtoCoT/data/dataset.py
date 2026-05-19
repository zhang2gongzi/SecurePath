"""
数据加载类
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class VulnerabilityDataset(Dataset):
    """漏洞检测数据集"""

    def __init__(self, csv_path, tokenizer_name='microsoft/codebert-base', max_length=512):
        """
        Args:
            csv_path: CSV文件路径
            tokenizer_name: tokenizer名称
            max_length: 最大序列长度
        """
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        print(f"加载数据: {csv_path}")
        print(f"样本数: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 获取代码和标签
        code = str(row['code'])
        label = int(row['label'])

        # tokenizer编码
        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'cwe_type': row.get('cwe_type', 'unknown')
        }


class ContrastiveDataset(Dataset):
    """对比学习数据集（正负样本对）"""

    def __init__(self, csv_path, tokenizer_name='microsoft/codebert-base', max_length=512):
        """
        Args:
            csv_path: CSV文件路径（包含vulnerable和fixed配对）
        """
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # 按原始样本配对（每两条：vulnerable + fixed）
        self.paired_data = []
        for i in range(0, len(self.data), 2):
            if i + 1 < len(self.data):
                self.paired_data.append({
                    'vulnerable': self.data.iloc[i]['code'],
                    'fixed': self.data.iloc[i + 1]['code'],
                    'cwe_type': self.data.iloc[i]['cwe_type']
                })

        print(f"加载数据: {csv_path}")
        print(f"配对样本数: {len(self.paired_data)}")

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, idx):
        pair = self.paired_data[idx]

        # 编码漏洞代码
        vul_encoding = self.tokenizer(
            str(pair['vulnerable']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 编码修复代码
        fix_encoding = self.tokenizer(
            str(pair['fixed']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'vul_input_ids': vul_encoding['input_ids'].squeeze(0),
            'vul_attention_mask': vul_encoding['attention_mask'].squeeze(0),
            'fix_input_ids': fix_encoding['input_ids'].squeeze(0),
            'fix_attention_mask': fix_encoding['attention_mask'].squeeze(0),
            'cwe_type': pair['cwe_type']
        }


def get_dataloaders(data_dir, tokenizer_name='microsoft/codebert-base', batch_size=16):
    """获取数据加载器"""
    from torch.utils.data import DataLoader

    # 训练集
    train_dataset = VulnerabilityDataset(
        f"{data_dir}/processed/train_contrastive.csv",
        tokenizer_name=tokenizer_name
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 验证集
    val_dataset = VulnerabilityDataset(
        f"{data_dir}/processed/val_ratio.csv",
        tokenizer_name=tokenizer_name
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 测试集
    test_dataset = VulnerabilityDataset(
        f"{data_dir}/processed/test_ratio.csv",
        tokenizer_name=tokenizer_name
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 测试数据加载
    print("测试数据加载...")

    data_dir = r'E:\paper\new\SecureProtoCoT\data'

    # 测试VulnerabilityDataset
    try:
        dataset = VulnerabilityDataset(f"{data_dir}/processed/train_contrastive.csv")
        print(f"数据集大小: {len(dataset)}")

        # 获取一个样本
        sample = dataset[0]
        print(f"样本形状: input_ids={sample['input_ids'].shape}, label={sample['label']}")
    except Exception as e:
        print(f"数据加载测试跳过（数据文件不存在）: {e}")
