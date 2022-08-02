import math
import torch
from torch import nn
from d2l import torch as d2l


def masked_softmax(X, valid_lens):
    """
    通过在最后一个维度上遮蔽元素执行softmax操作
    :param X: 3D的张量
    :param valid_lens: 有效长度，1D or 2D
    :return: 遮蔽的softmax
    """
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一个维度用大的负值代替，使其softmax后输出为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=1e-6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    """加性注意力，一般用于query和key维度不一样的情况"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # 此时queries的维度为(batch_size,查询个数,num_hiddens)
        # keys的维度为(batch_size,键值对个数,num_hiddens)
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 使用广播方式求和,1的那些维度都自动扩展
        features = torch.tanh(queries.unsqueeze(2) + keys.unsqueeze(1))
        # 移除最后一个1的维度，故score:(batch_size,查询个数,键值对个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values:(batch_size,键值对个数,值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """缩放点积注意力，一般用于query和keys维度一致的情况"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        :param queries: (batch_size,查询个数,d)
        :param keys: (batch_size,键值对个数,d)
        :param values: (batch_size,键值对个数,v)
        :param valid_lens: (batch_size,)or(batch_size,查询个数),因为查询一般是当前的输入，
                            所以第二个维度可能是查询维度
        :return:
        """
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


if __name__ == "__main__":
    queries = torch.normal(0, 1, (2, 1, 2))
    keys = torch.normal(0, 1, (2, 2, 2))
    values = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 3])
    dot_attention = DotProductAttention(dropout=0.5)
    add_attention = AdditiveAttention(2, 2, 8, 0.5)
    add_attention.eval()
    add_attention(queries, keys, values, valid_lens)
    dot_attention.eval()
    dot_attention(queries, keys, values, valid_lens)
    d2l.show_heatmaps(add_attention.attention_weights.reshape(1, 1, 2, 2), figsize=(10, 10),
                      xlabel="keys", ylabel="queries", titles="Add attention")
    d2l.show_heatmaps(dot_attention.attention_weights.reshape(1, 1, 2, 2), figsize=(10, 10),
                      xlabel="keys", ylabel="queries", titles="Dot attention")
    d2l.plt.show()

