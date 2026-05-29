#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可训练 Attention Pooling 模块
替代固定的 CLS pooling，让模型学会关注与安全性相关的 token
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    可训练的注意力池化层

    对 CodeBERT 输出的全部 token hidden states 学习注意力权重，
    加权求和得到固定维度表示。自动遮罩 padding token。

    Args:
        hidden_dim: CodeBERT hidden size (768 for base)
    """
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim) - 全token的last_hidden_state
            attention_mask: (batch, seq_len) - 1=real token, 0=padding
        Returns:
            pooled: (batch, hidden_dim) - 注意力加权后的表示
        """
        scores = self.query(hidden_states).squeeze(-1)          # (B, L)
        scores = scores.masked_fill(attention_mask == 0, -1e9)  # 遮罩 padding
        weights = F.softmax(scores, dim=-1)                     # (B, L)
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)  # (B, H)
        return pooled


class SafetyClassifier(nn.Module):
    """
    安全分类器：AttentionPooling + MLP

    输入：CodeBERT 的 last_hidden_state + attention_mask
    输出：P(vul | code)
    """
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.attention_pool = AttentionPooling(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, hidden_states, attention_mask):
        pooled = self.attention_pool(hidden_states, attention_mask)
        return self.mlp(pooled)
