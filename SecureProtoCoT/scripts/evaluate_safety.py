#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验A：MSR测试集自动评估 + 验证集一致性检查
加载编码器 + 原型，在test_contrastive.csv和val_contrastive.csv上评估
"""
import os, sys, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path

# ================= 配置 =================
CONFIG = {
    'model_path': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/models/best_model',
    'data_dir': '/home2/zzl/SecurePath/SecureProtoCoT/data/processed',
    'proto_dir': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/models',
    'output_dir': '/home2/zzl/SecurePath/SecureProtoCoT/outputs/results',
    'max_length': 512,
    'batch_size': 32,
    'projection_dim': 256,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# ================= 编码器（与训练时结构一致） =================
class ContrastiveEncoder(torch.nn.Module):
    def __init__(self, model_path, projection_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        self.hidden_size = self.encoder.config.hidden_size
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        projection = self.projection(cls_embedding)
        return torch.nn.functional.normalize(projection, p=2, dim=1)

# ================= 数据加载 =================
class EvalDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.codes = df['code'].astype(str).tolist()
        self.labels = df['label'].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.codes)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.codes[idx], max_length=self.max_length,
                             padding='max_length', truncation=True)
        return {
            'input_ids': torch.tensor(enc['input_ids']),
            'attention_mask': torch.tensor(enc['attention_mask']),
            'labels': torch.tensor(self.labels[idx])
        }

# ================= 评估函数 =================
def run_eval(name, df, model, tokenizer, safe_proto, vul_proto, device, max_length, batch_size, output_dir):
    """对单个数据集运行评估，返回指标字典"""
    print(f"\n{'=' * 60}")
    print(f"[{name}] 样本: 总计 {len(df)}, safe={len(df[df['label']==0])}, vul={len(df[df['label']==1])}")
    print(f"{'=' * 60}")

    ds = EvalDataset(df, tokenizer, max_length)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    all_labels, all_preds, all_scores = [], [], []

    with torch.no_grad():
        for batch in tqdm(dl, desc=f"[{name}]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            embeddings = model(input_ids, attention_mask)
            sim_safe = torch.nn.functional.cosine_similarity(embeddings, safe_proto.unsqueeze(0), dim=1)
            sim_vul = torch.nn.functional.cosine_similarity(embeddings, vul_proto.unsqueeze(0), dim=1)
            # 越接近漏洞原型 → sim_vul 越大 → scores > 0 → pred=1(漏洞)
            scores = sim_vul - sim_safe
            preds = (scores >= 0).long().cpu().numpy()

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_scores.extend(scores.cpu().tolist())

    metrics = {
        'name': name,
        'total': len(all_labels),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_scores),
    }

    # 得分分布
    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)
    metrics['safe_mean'] = scores_arr[labels_arr == 0].mean()
    metrics['vul_mean'] = scores_arr[labels_arr == 1].mean()

    # 保存结果
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame({'label': all_labels, 'pred': all_preds, 'score': all_scores})
    out_path = os.path.join(output_dir, f'msr_eval_{name}.csv')
    result_df.to_csv(out_path, index=False)

    return metrics

# ================= 主流程 =================
def main():
    print("=" * 60)
    print("实验A：MSR测试集自动评估 + 验证集一致性检查")
    print("=" * 60)
    print(f"设备: {CONFIG['device']}")

    # 1. 加载模型
    print(f"\n加载编码器: {CONFIG['model_path']}")
    model = ContrastiveEncoder(CONFIG['model_path'], projection_dim=CONFIG['projection_dim'])
    model = model.to(CONFIG['device'])

    # 加载完整模型参数（编码器+投影头）
    state_dict_path = os.path.join(CONFIG['output_dir'], 'model_state.pt')
    if os.path.exists(state_dict_path):
        model.load_state_dict(torch.load(state_dict_path, map_location=CONFIG['device']))
        print("已加载完整模型参数（含投影头）")
    else:
        print("⚠️ 未找到完整模型参数，投影头为随机初始化")

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_path'])

    # 2. 加载原型
    safe_proto = torch.load(os.path.join(CONFIG['proto_dir'], 'safe_prototype.pt'), map_location=CONFIG['device'])
    vul_proto = torch.load(os.path.join(CONFIG['proto_dir'], 'vul_prototype.pt'), map_location=CONFIG['device'])
    print(f"  安全原型: {safe_proto.shape}, 漏洞原型: {vul_proto.shape}")

    # 3. 加载数据
    test_df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'test_contrastive.csv'))
    val_df  = pd.read_csv(os.path.join(CONFIG['data_dir'], 'val_contrastive.csv'))

    # 4. 分别在 test 和 val 上评估
    test_metrics = run_eval('test', test_df, model, tokenizer, safe_proto, vul_proto,
                            CONFIG['device'], CONFIG['max_length'], CONFIG['batch_size'], CONFIG['output_dir'])
    val_metrics  = run_eval('val',  val_df,  model, tokenizer, safe_proto, vul_proto,
                            CONFIG['device'], CONFIG['max_length'], CONFIG['batch_size'], CONFIG['output_dir'])

    # 5. 对比报告
    print(f"\n{'=' * 60}")
    print("评估结果汇总")
    print(f"{'=' * 60}")

    header = f"{'指标':<18} {'Test集':>10} {'Val集':>10} {'差异':>10} {'一致性':>10}"
    print(header)
    print("-" * len(header))

    keys = [
        ('样本数', 'total'),
        ('Accuracy', 'accuracy'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('F1', 'f1'),
        ('AUC', 'auc'),
        ('安全代码均分', 'safe_mean'),
        ('漏洞代码均分', 'vul_mean'),
    ]

    all_ok = True
    for label, key in keys:
        tv = test_metrics[key]
        vv = val_metrics[key]
        diff = abs(tv - vv)

        if key == 'total':
            ok = True  # 样本数无需一致性判断
            mark = ''
        else:
            ok = diff < 0.05  # 差异小于 5% 视为一致
            mark = '✅' if ok else '⚠️'
            if not ok: all_ok = False

        print(f"{label:<18} {tv:>10.4f} {vv:>10.4f} {diff:>10.4f} {mark:>10}")

    # 6. 一致性判断
    print(f"\n{'=' * 60}")
    if all_ok:
        print("✅ 一致性检查通过: Test 和 Val 指标差异均 < 0.05")
    else:
        print("⚠️ 一致性检查未通过: 存在指标差异 >= 0.05，请排查")

    # 7. 实验A 达标判断（以 Test 集为准）
    f1_test = test_metrics['f1']
    auc_test = test_metrics['auc']
    print(f"\n{'=' * 60}")
    if f1_test > 0.75 and auc_test > 0.85:
        print(f"✅ 实验A 达标 (Test): F1={f1_test:.4f} > 0.75, AUC={auc_test:.4f} > 0.85")
    else:
        print(f"⚠ 实验A 未达标 (Test): F1={f1_test:.4f} (需>0.75), AUC={auc_test:.4f} (需>0.85)")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
