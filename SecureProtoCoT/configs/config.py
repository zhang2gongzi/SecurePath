"""
SecureProtoCoT 配置文件
"""

# 数据配置
DATA_CONFIG = {
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
    'samples_per_cwe': 1000,

    # 数据划分比例
    'train_ratio': 0.7,
    'val_ratio': 0.1,
    'test_ratio': 0.2,

    # 代码长度过滤
    'min_lines': 10,
    'max_lines': 200,
}

# 模型配置
MODEL_CONFIG = {
    'encoder_name': 'microsoft/codebert-base',  # 或 'microsoft/graphcodebert-base'
    'max_length': 512,
    'hidden_size': 768,
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 10,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'random_seed': 42,
}

# 对比学习配置
CONTRASTIVE_CONFIG = {
    'temperature': 0.07,
    'projection_dim': 256,
}

# 路径配置
PATH_CONFIG = {
    'data_dir': r'E:\paper\new\SecureProtoCoT\data',
    'output_dir': r'E:\paper\new\SecureProtoCoT\outputs',
    'model_dir': r'E:\paper\new\SecureProtoCoT\outputs\models',
}
