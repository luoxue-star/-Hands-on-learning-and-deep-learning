import math
import torch
from torch import nn
from d2l import torch as d2l


class PositionalEncoding(nn.Module):
    """自注意力位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个batch为1，足够长的长度*隐藏维度的位置编码
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)  # 偶数列为sin，奇数列为cos
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        # X的维度是(batch_size,序列长度，特征维度)
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

